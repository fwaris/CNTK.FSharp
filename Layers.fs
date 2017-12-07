namespace CNTKWrapper
open CNTK
open System
open Blocks
open FsBase

//based on python layers module (see CNTK Python API for documentation)
//mimics python code closely

module Layers =

    let dataType = DataType.Float           //default data type  TODO: make configurable
    let device = DeviceDescriptor.GPUDevice(0) //default device

    let inline private (!!) (v:Option<_>) = defaultArg v null

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

    let varVector (vs:Variable seq) =
        let vv = new VariableVector(Seq.length vs)
        vs |>  Seq.iter vv.Add
        vv

    let intVector (is:int seq) =
        let vs = new IntVector(Seq.length is)
        is |>  Seq.iter vs.Add
        vs

    type Activation = 
        | NONE
        | ReLU
        | Sigmoid
        | Tanh
        | LeakyReLU
        | PReLU of float

    let inline asList sz x  = [for _ in 1 .. sz -> x]

    //usually, much shape manipulation is done - so a separate type
    type Shape = D (*size of dimension*) of int | Ds of int list | Unspecified 

    let fromNDShape (s:NDShape) = s.Dimensions |> Seq.toList |> Ds
    let ( !+ ) (s:NDShape) = fromNDShape s
    let toNDShape = function D i -> shape [i] | Ds ds -> shape ds | Unspecified -> null
    let ( !- ) s = toNDShape s
    let dims = function D i -> [i] | Ds is -> is | Unspecified -> failwith "unspecified shape"
    let len = function D i -> 1 | Ds is -> List.length is | Unspecified -> 0

    //Shape operations
    type Shape with 
        static member ( + ) (s1:Shape,s2:Shape) =
            match s1,s2 with
            | D i, D j -> Ds [i; j]
            | D i, Ds js -> List.append js [i] |> Ds
            | Ds is, D j -> List.append is [j] |> Ds
            | Ds is, Ds js -> List.append is js |> Ds
            | Unspecified,_ 
            | _, Unspecified -> failwith "unspecified shape"

        static member ( + ) (s1:Shape,d:int) =
            match s1 with
            | D i   -> Ds [i; d]
            | Ds is -> List.append is [d] |> Ds
            | Unspecified -> failwith "unspecified shape"

        static member ( * )  (x:Shape, repeat:int) =
            match x with
            | D i -> Ds [for _ in 1 .. repeat -> i]
            | Ds is -> List.collect yourself [for _ in 1 .. repeat -> is] |> Ds
            | Unspecified -> failwith "unspecified shape"

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
            | Activation.PReLU c    -> let alpha = new Constant(v.Output.Shape, dataType, c)
                                       C.PReLU(!>v, alpha)

        static member private _window (x:Variable, axis, _begin, _end, step, stride, ?initial_state) = 
            if stride <> 1 then failwith "windowed convolution with stride not yet implemented"
            let initial_state = initial_state |> Option.defaultValue null
            let shifted =
                [|
                    for t in _begin .. step .. _end do
                        yield 
                            match t with
                            | 0             -> x
                            | t when t < 0 -> !> C.PastValue(x,initial_state, uint32 -t)
                            | t            -> !> C.FutureValue(x,initial_state,uint32 t)
                |]
            C.Splice(varVector shifted, axis)

        
        static member Dense 
            (
                x : Variable,
                output_shape,
                ?activation,
                ?init,
                ?input_rank,
                ?map_rank,
                ?bias,
                ?init_bias,
                ?name
            ) =
            let activation = defaultArg activation Activation.NONE
            let init = defaultArg init (C.GlorotUniformInitializer())
            let bias = defaultArg bias true
            let init_bias = defaultArg init_bias 0.
            let name = defaultArg  name ""

            let infer_input_rank_to_map =
                match input_rank,map_rank with
                | Some _, Some _    -> failwith "Dense: input_rank and map_rank cannot be specified at the same time."
                | Some _, None      -> -1
                | _     , None      -> 0
                | _     , Some r    -> r

            let output_rank = len output_shape
            let input_shape = D NDShape.InferredDimension * (match input_rank with None -> 1 | Some r -> r)
            
            let init_weight = B._initializer_with_rank (init, output_rank=output_rank) 
            let W = new Parameter(!-(input_shape + output_shape),dataType,init_weight,device,"W")
            let b = new Parameter(!-output_shape,dataType,init_bias,device,"b")

            let r = C.Times(x,W,uint32 output_rank, infer_input_rank_to_map)
            let r = if bias then C.Plus(!>r,  b ) else r
            let r = L.activation r activation
            r

                                 
        static member Convolution
            (
                convVar: Variable,
                filter_shape,
                ?num_filters,
                ?sequential,
                ?activation,
                ?init,
                ?pad,
                ?strides,
                ?sharing,
                ?bias,
                ?init_bias,
                ?reduction_rank,
                ?transpose_weight,
                ?dialation,
                ?max_temp_mem_size_in_samples,
                ?op_name,
                ?name
            ) =
            let num_filters = defaultArg num_filters 0
            let sequential = defaultArg sequential false 
            let activation = defaultArg activation Activation.NONE
            let init = defaultArg init (C.GlorotUniformInitializer())
            let pad = defaultArg pad false
            let strides = defaultArg strides (D 1)
            let sharing = defaultArg sharing true
            let bias = defaultArg bias true
            let init_bias = defaultArg init_bias 0.
            let reduction_rank = defaultArg reduction_rank 1
            let transpose_weight = defaultArg transpose_weight false
            let dialation = defaultArg dialation (D 1)
            let max_temp_mem_size_in_samples = defaultArg max_temp_mem_size_in_samples 0
            let op_name = defaultArg op_name "Convolution"
            let name = defaultArg  name ""

            if [0;1] |> List.contains reduction_rank |> not then
                failwith "Convolution: reduction_rank must be 0 or 1"
            if transpose_weight then
                failwith "Convolution: transpose_weight option currently not supported"
            if not sharing then
                failwith "Convolution: sharing option currently must be True"

            let num_filters = if num_filters = 0 then Ds [] else D num_filters
            let filter_rank = len filter_shape
            let strides = strides .padTo filter_shape
            let sharing = asList filter_rank sharing 
            let pad     = asList filter_rank pad
            let dialation = dialation .padTo filter_shape

            let emulating_output_depth = len num_filters = 0
            let emulating_input_depth = reduction_rank = 0

            let actual_output_channels_shape = 
                if not emulating_output_depth then
                    num_filters
                else
                    D 1

            let actual_reduction_shape = D NDShape.InferredDimension 
            let actual_filter_shape = filter_shape

            let num_emulated_axes = if emulating_input_depth then 1 else 0
            let strides = (D 1) * num_emulated_axes + strides
            let sharing = asList num_emulated_axes true @ sharing
            let pad = asList num_emulated_axes false @ pad

            let kernel_shape = actual_reduction_shape + actual_filter_shape

            //simplified version of python code which I
            //don't fully understand yet
            let init_kernel = B._initializer_with_rank(
                                init,
                                filter_rank = filter_rank,
                                output_rank = -len(actual_output_channels_shape)
                                )

            let W = new Parameter(
                        !-(actual_output_channels_shape + kernel_shape),
                        dataType,
                        init_kernel,
                        device,
                        "W")

            let b = if bias then
                        new Parameter(
                            !-(actual_output_channels_shape + (D 1) * len(actual_filter_shape)),
                            dataType,
                            init_bias,
                            device,
                            "b")
                        else
                            null
            
            let filter_rank_without_seq = if sequential then filter_rank - 1 else filter_rank
            let num_inserted_axes = if sequential then 1 + num_emulated_axes else num_emulated_axes

            let beginAxis = 
                if filter_rank_without_seq <> 0 then 
                    new Axis(-filter_rank_without_seq)
                else
                    Axis.EndStaticAxis() //python code's Axis.new_leading_axis() resolves to this

            let endAxis = 
                if filter_rank_without_seq <> 0 then
                    new Axis(-filter_rank_without_seq)
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

            let rank1 = (dims filter_shape |> List.rev).[filter_rank-1] //filter_shape[-filter_rank] in python
            let lpad = (rank1 - 1) / 2
            let x = 
                if sequential then
                    let stride1 = (dims strides |> List.rev).[filter_rank-1]
                    L._window(!>x,new Axis(-filter_rank),-lpad,-lpad+rank1,1,stride1)
                else
                    x

            let sequential_emulated_axis = if sequential then pad.Length - filter_rank |> Some else None
            let isEmulated n = match sequential_emulated_axis with Some y -> y = n | None -> false
            let autoPadding =  
                asList reduction_rank false 
                @ 
                (pad |> List.mapi (fun i p -> if isEmulated i |> not then p else false))

            let r = C.Convolution(
                        W,
                        !>x,
                        !-strides,
                        boolVector sharing,
                        boolVector autoPadding,
                        !-dialation,
                        uint32 reduction_rank,
                        uint32 max_temp_mem_size_in_samples,
                        "convolution")
            
            let zeroPad = (pad |> List.rev).[filter_rank - 1] 
            let r = 
                let begin_index = intVector [lpad]
                let end_index = intVector[-(rank1-1-lpad)]
                if sequential && not zeroPad then
                    C.Slice(!>r, null, begin_index , end_index)
                else
                    r

            let r = if bias then C.Plus(!>r,b) else r

            let num_axes_to_remove = [sequential; emulating_output_depth] |> List.map (function true -> 1 | false -> 0) |> List.sum
            let r = 
                if num_axes_to_remove > 0 then
                    let begin_axis = new Axis(-filter_rank_without_seq - num_axes_to_remove)
                    let end_axis = if filter_rank_without_seq <> 0 then new Axis(-filter_rank_without_seq) else null
                    C.Reshape(!>r, !- (Ds []),  begin_axis, end_axis)
                else
                    r
            let r = L.activation r activation

            r

        static member Convolution2D
            (
                convVar: Variable,
                filter_shape,
                ?num_filters,
                ?activation,
                ?init,
                ?pad,
                ?strides,
                ?bias,
                ?init_bias,
                ?reduction_rank,
                ?dialation,
                ?name
            ) =
                if len(filter_shape) > 2 then failwith "Convolution2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)"
                let filter_shape = filter_shape .padTo (Ds [0;0])
                L.Convolution(
                    convVar,
                    filter_shape,
                    num_filters=defaultArg num_filters 0,
                    activation=defaultArg activation Activation.NONE,
                    init=defaultArg init (C.GlorotUniformInitializer()),
                    pad=defaultArg pad false,
                    strides=defaultArg strides (D 1),
                    bias=defaultArg bias true,
                    init_bias=defaultArg init_bias 0.,
                    reduction_rank=defaultArg reduction_rank 1,
                    dialation=defaultArg dialation (D 1),
                    name=defaultArg name "")

        static member BatchNormalization
            (
                x: Variable,
                ?map_rank,
                ?init_scale,
                ?normalization_time_constant,
                ?blend_time_constant,
                ?epsilon,
                ?use_cntk_engine,
                ?disable_regularization,
                ?name
            ) = 

            let map_rank =
                match map_rank with
                | None   -> 0
                | Some 1 -> 1
                | Some x -> failwith "map_rank can only be null or 1 for now"

            let normalization_time_constant = defaultArg normalization_time_constant 5000
            let blend_time_constant         = defaultArg blend_time_constant  0
            let epsilon                     = defaultArg epsilon  0.00001
            let use_cntk_engine             = defaultArg use_cntk_engine  false
            let init_scale                  = defaultArg init_scale 1.0

            let norm_shape = !- (D NDShape.InferredDimension)

            let scale        = new Parameter(norm_shape, dataType, init_scale, device, "scale")
            let bias         = new Parameter(norm_shape, dataType, 0., device, "bias")
            let run_mean     = new Constant( norm_shape, dataType, 0., device, "aggregate_mean")
            let run_variance = new Constant( norm_shape, dataType, 0., device, "aggregate_variance")
            let run_count    = new Constant( !-(Ds []) , dataType, 0., device, "aggregate_count")

            C.BatchNormalization(
                x,
                scale,
                bias,
                run_mean,
                run_variance,
                run_count,
                (map_rank = 1),
                float normalization_time_constant,
                float blend_time_constant,
                epsilon,
                not use_cntk_engine,
                name = "batch_normalization"
            )

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
    
        static member ConvolutionTranspose 
            (
                convVar : Variable,
                filter_shape, 
                ?num_filters,
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
        
            let num_filters = defaultArg num_filters 0 //probably not correct as python defaults to null
            let activation = defaultArg activation Activation.NONE
            let init = defaultArg init (C.GlorotUniformInitializer())
            let pad = defaultArg pad false
            let strides = defaultArg strides (D 1)
            let sharing = defaultArg sharing true
            let bias = defaultArg bias true
            let init_bias = defaultArg init_bias 0.
            let reduction_rank = defaultArg reduction_rank 1
            let dialation = defaultArg dialation (D 1)
            let max_temp_mem_size_in_samples = defaultArg max_temp_mem_size_in_samples 0
            let name = defaultArg  name ""

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
                | None | Some Shape.Unspecified -> output_channels_shape
                | Some (osp:Shape) -> output_channels_shape + osp

            let filter_rank = len filter_shape
            let init_kernel = B._initializer_with_rank (init, filter_rank = filter_rank, output_rank = -1)

            let W = new Parameter(!-kernel_shape, dataType, init_kernel,device,"W")
            let b = 
                if bias then
                    new Parameter (
                        (output_channels_shape + 1) * filter_rank |> toNDShape, 
                         dataType, 
                         init_bias,
                         device,
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
                ?num_filters,
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
               
            if len filter_shape  > 2 then            
                failwith "ConvolutionTranspose2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)"

            let filter_shape = filter_shape .padTo (Ds [0;0])

            L.ConvolutionTranspose(
                convVar,
                filter_shape,
                num_filters=defaultArg num_filters 0,
                activation=defaultArg activation Activation.NONE,
                init=defaultArg init (C.GlorotUniformInitializer()),
                pad=defaultArg pad false,
                strides=defaultArg strides (D 1),
                bias=defaultArg bias true,
                init_bias=defaultArg init_bias 0.,
                reduction_rank=defaultArg reduction_rank 1,
                dialation=defaultArg dialation (D 1),
                name=defaultArg name "")
