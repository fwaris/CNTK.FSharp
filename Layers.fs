namespace CNTKWrapper
open CNTK
open System
type C = CNTKLib
open Blocks
open FsBase
//based on python layers module (see CNTK Python API for documentation)

module Layers =

    let dataType = DataType.Float           //default data type  TODO: make configurable
    let gpu = DeviceDescriptor.GPUDevice(0) //default device

    let scalar x = Constant.Scalar(dataType,x)
    let shape (dims:int seq) = NDShape.CreateNDShape dims

    let inline idict (s:(^a * ^b) seq) =
        let d = new System.Collections.Generic.Dictionary< ^a, ^b>()
        s |> Seq.iter d.Add
        d

    let parmVector (ps:Parameter seq) = 
        let pv = new ParameterVector(Seq.length ps)
        ps |>  Seq.iter pv.Add
        pv

    let lrnVector (ls:Learner seq) =
        let lv = new LearnerVector(Seq.length ls)
        ls |>  Seq.iter lv.Add
        lv

    let boolVector (ls:bool seq) =
        let lv = new BoolVector(Seq.length ls)
        ls |>  Seq.iter lv.Add
        lv

    let prgwVector (pws:ProgressWriter seq) =
        let pwv = new ProgressWriterVector(Seq.length pws)
        pws |>  Seq.iter pwv.Add
        pwv

    type Activation = 
        | NONE
        | ReLU
        | Sigmoid
        | Tanh
        | LeakyReLU

    let inline asList sz x  = [for _ in 1 .. sz -> x]

    //usually, much shape manipulation is done - so a separate type
    type Shape = D of int | Ds of int list

    let fromNDShape (s:NDShape) = s.Dimensions |> Seq.toList |> Ds
    let ( !+ ) (s:NDShape) = fromNDShape s
    let toNDShape = function D i -> shape [i] | Ds ds -> shape ds
    let ( !- ) s = toNDShape s
    let dims = function D i -> [i] | Ds is -> is
    let len = function D i -> 1 | Ds is -> List.length is

    //Shape operations
    type Shape with 
        static member ( + ) (s1:Shape,s2:Shape) =
            match s1,s2 with
            | D i, D j -> Ds [i; j]
            | D i, Ds js -> List.append js [i] |> Ds
            | Ds is, D j -> List.append is [j] |> Ds
            | Ds is, Ds js -> List.append is js |> Ds

        static member ( + ) (s1:Shape,d:int) =
            match s1 with
            | D i   -> Ds [i; d]
            | Ds is -> List.append is [d] |> Ds

        static member ( * )  (x:Shape, repeat:int) =
            match x with
            | D i -> Ds [for _ in 1 .. repeat -> i]
            | Ds is -> List.collect yourself [for _ in 1 .. repeat -> is] |> Ds

        member x.padTo (s2:Shape) =
            match x,s2 with
            | D i, D j -> D i
            | D i, Ds js -> js |> List.map (fun  _ -> i) |> Ds
            | Ds is, Ds js when is.Length=js.Length -> x
            | _,_ -> failwithf "shape must be singular or the dimensions should match s2"


    //layers type
    type L =
        static member  activation v = function
            | Activation.NONE       ->              v
            | Activation.ReLU       -> C.ReLU       !>v
            | Activation.LeakyReLU  -> C.LeakyReLU  !>v
            | Activation.Sigmoid    -> C.Sigmoid    !>v
            | Activation.Tanh       -> C.Tanh       !>v

        static member FullyConnectedLinearLayer
            (
                input:Variable, 
                outputDim:Shape, 
                init: CNTKDictionary,
                device:DeviceDescriptor,
                outputName:string
            ) : Function =

            let inputDim = input.Shape.[0]

            let timesParam = 
                let parmShape = 
                   !- (outputDim  + !+ input.Shape)
                new Parameter(
                    parmShape,
                    DataType.Float,
                    init,
                    device, 
                    "timesParam")

            let timesFunction = 
                new Variable(C.Times(timesParam, input, "times"))

            let plusParam = new Parameter( !- outputDim, 0.0f, device, "plusParam")
            C.Plus(plusParam, timesFunction, outputName)

        static member Dense(
                            input:Variable, 
                            outputDim:Shape,
                            device:DeviceDescriptor,
                            init:CNTKDictionary,
                            activation:Activation, 
                            outputName:string) : Function =

                let input : Variable =
                    if (input.Shape.Rank <> 1)
                    then
                        let newDim = input.Shape.Dimensions |> Seq.reduce(fun d1 d2 -> d1 * d2)
                        new Variable(C.Reshape(input, shape [ newDim ]))
                    else input

                let fullyConnected : Function = 
                    L.FullyConnectedLinearLayer(input, outputDim, init, device, outputName)

                L.activation fullyConnected activation
    
        static member ConvolutionTranspose 
            (
                convVar : Variable,
                filter_shape, 
                num_filters,
                ?activation,
                ?init,
                ?pad,
                ?strides,
                ?sharing,
                ?bias,
                ?init_bias,
                ?output_shape,
                ?reduction_rank,
                ?dialation,
                ?max_temp_mem_size_in_samples,
                ?name
            ) = 
        
            let activation =  activation |> Option.defaultValue Activation.NONE
            let init =  init |> Option.defaultValue (C.GlorotUniformInitializer())
            let pad  = pad |> Option.defaultValue false
            let strides = strides |> Option.defaultValue (D 1)
            let sharing = sharing |> Option.defaultValue true
            let bias = bias |> Option.defaultValue true
            let init_bias = init_bias |> Option.defaultValue 0.
            //output_shape : no default value
            let reduction_rank = reduction_rank |> Option.defaultValue 1
            let dialation = dialation |> Option.defaultValue (D 1)
            let max_temp_mem_size_in_samples = max_temp_mem_size_in_samples |> Option.defaultValue 0
            let name= name |> Option.defaultValue ""

            if [0;1] |> List.contains reduction_rank |> not then
                failwith "ConvolutionTranspose: reduction_rank must be 0 or 1"
            if not sharing then 
                failwith "ConvolutionTranspose: sharing option currently must be true"

            //tuplify all tuple inputs that can also be given as scalars if rank 1
            //filter_shape = already given as Shape
            let num_filters  = D num_filters
            let strides      = strides .padTo filter_shape
            let sharing      = asList (len filter_shape) sharing
            let pad          = asList (len filter_shape) pad 
            let dialation    = dialation .padTo filter_shape 

            let emulating_input_depth = if reduction_rank = 0 then 1 else 0

            let num_emulated_axes = emulating_input_depth
            let strides = (D 1) * num_emulated_axes + strides
            let sharing = asList num_emulated_axes true @ sharing |> boolVector
            let pad     = asList num_emulated_axes false @ pad    
            let autoPadding = asList reduction_rank false @ pad |> boolVector
            let output_channels_shape = num_filters

            let kernel_shape = 
                D NDShape.InferredDimension
                + output_channels_shape
                + filter_shape

            let output_full_shape = 
                match output_shape with
                | None -> output_channels_shape
                | Some (osp:Shape) -> output_channels_shape + osp

            let filter_rank = len filter_shape
            let init_kernel = B._initializer_with_rank (init, filter_rank = filter_rank, output_rank = -1)

            let W = new Parameter(!-kernel_shape, dataType, init_kernel,gpu,"W")
            let b = 
                if bias then
                    new Parameter (
                        (output_channels_shape + 1) * filter_rank |> toNDShape, 
                         dataType, 
                         init_bias,
                         gpu,
                         "b")
                     |> Some
                else
                    None

            let num_inserted_axes = num_emulated_axes

            let beginAxis = 
                if filter_rank <> 0 then 
                    new Axis(-filter_rank)
                else
                    Axis.EndStaticAxis() //python code's Axis.new_leading_axis() resolves to this

            let endAxis = 
                if filter_rank <> 0 then
                    new Axis(-filter_rank)
                else
                    null

            let x = 
                if num_inserted_axes <> 0 then
                    C.Reshape (
                        convVar, 
                        (D 1) * num_inserted_axes |> toNDShape,
                        beginAxis,
                        endAxis
                        )
                else
                    !> convVar

            let r = 
                C.ConvolutionTranspose (
                    W,
                    !> x,
                    !-strides,
                    sharing,
                    autoPadding,
                    !-output_full_shape,
                    !-dialation,
                    uint32 max_temp_mem_size_in_samples
                )

            let r = match b with Some b -> C.Plus(!>r,b) | None -> r

            L.activation r activation


    //ConvolutionTranspose2D -- create a 2D convolution transpose layer with optional non-linearity
        static member ConvolutionTranspose2D 
            (
                convVar : Variable,
                filter_shape : Shape, //a 2D tuple, e.g., (3,3),
                num_filters,
                ?activation,
                ?init,
                ?pad,
                ?strides,
                ?bias,
                ?init_bias,
                ?output_shape,
                ?reduction_rank,
                ?dialation,
                ?name
            ) = 
        
            let activation =  activation |> Option.defaultValue Activation.NONE
            let init =  init |> Option.defaultValue (C.GlorotUniformInitializer())
            let pad  = pad |> Option.defaultValue false
            let strides = strides |> Option.defaultValue (D 1)
            let bias = bias |> Option.defaultValue true
            let init_bias = init_bias |> Option.defaultValue 0.
            //output_shape : no default value
            let reduction_rank = reduction_rank |> Option.defaultValue 1
            let dialation = dialation |> Option.defaultValue (D 1)
            let name= name |> Option.defaultValue ""
       
            if len filter_shape  > 2 then            
                failwith "ConvolutionTranspose2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)"

            let filter_shape = filter_shape .padTo (Ds [0;0])

            match output_shape with
            | None ->
                L.ConvolutionTranspose(
                    convVar,
                    filter_shape,
                    num_filters,
                    activation,
                    init,
                    pad,
                    strides,
                    true,
                    bias,
                    init_bias,
                    //output_shape=osp,
                    reduction_rank=reduction_rank,
                    dialation=dialation,
                    name=name)
            | Some osp -> 
                L.ConvolutionTranspose(
                    convVar,
                    filter_shape,
                    num_filters,
                    activation,
                    init,
                    pad,
                    strides,
                    true,
                    bias,
                    init_bias,
                    output_shape=osp,
                    reduction_rank=reduction_rank,
                    dialation=dialation,
                    name=name)
