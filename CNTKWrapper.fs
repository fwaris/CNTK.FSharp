module CNTKWrapper
open CNTK
open System

type C = CNTKLib

//utility operator for F# implicit conversions 
let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

let dataType = DataType.Float           //default data type
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

//usually, much shape manipulation is done - so a separate type
type Shape = D of int | Ds of int list

let ( ++ ) (s1:Shape) (s2:Shape) =
    match s1,s2 with
    | D i, D j -> Ds [i; j]
    | D i, Ds js -> List.append js [i] |> Ds
    | Ds is, D j -> List.append is [j] |> Ds
    | Ds is, Ds js -> List.append is js |> Ds

let fromNDShape (s:NDShape) = s.Dimensions |> Seq.toList |> Ds
let ( !+ ) (s:NDShape) = fromNDShape s
let toNDShape = function D i -> shape [i] | Ds ds -> shape ds
let ( !- ) s = toNDShape s
let dims = function D i -> [i] | Ds is -> is

type Shape with 
    member x.padTo (s2:Shape) =
        match x,s2 with
        | D i, D j -> D i
        | D i, Ds js -> js |> List.map (fun  _ -> i) |> Ds
        | Ds is, Ds js when is.Length=js.Length -> x
        | _,_ -> failwithf "shape must be singular or the dimensions should match s2"

//layers type
type L =

    static member FullyConnectedLinearLayer(
                                                input:Variable, 
                                                outputDim:Shape, 
                                                init: CNTKDictionary,
                                                device:DeviceDescriptor,
                                                outputName:string) : Function =

        let inputDim = input.Shape.[0]

        let timesParam = 
            let parmShape = 
               !- (outputDim  ++ !+ input.Shape)
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
    
            match activation with
            | Activation.NONE       -> fullyConnected
            | Activation.ReLU       -> C.ReLU       !>fullyConnected
            | Activation.LeakyReLU  -> C.LeakyReLU  !>fullyConnected
            | Activation.Sigmoid    -> C.Sigmoid    !>fullyConnected
            | Activation.Tanh       -> C.Tanh       !>fullyConnected

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
            ?name
        ) = 
        
        let activation =  activation |> Option.defaultValue Activation.None
        let init =  init |> Option.defaultValue (C.GlorotUniformInitializer())
        let pad  = pad |> Option.defaultValue false
        let bias = bias |> Option.defaultValue true
        let init_bias = init_bias |> Option.defaultValue 0.f
        let strides = strides |> Option.defaultValue (D 1)
        let name= name |> Option.defaultValue ""
        let sharing = sharing |> Option.defaultValue true
        let output_channels_shape = num_filters

        let kernel_shape = 
            D NDShape.InferredDimension
            ++ output_channels_shape
            ++ filter_shape

        let output_full_shape = 
            match output_shape with
            | None -> output_channels_shape
            | Some osp -> output_channels_shape ++ osp

        let strides = strides .padTo filter_shape
        let sharing = dims strides |> List.map (fun _ -> sharing) |> boolVector
        let autoPadding = boolVector [pad] 

        C.ConvolutionTranspose
            (
                W,
                x,
                !-strides,
                sharing,
                autoPadding,
                !-output_full_shape
            )


//based from CNTK Python implementation of same name (see CNTK repo)
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
            ?name
        ) = 
        
        let num_filters = num_filters |> Option.defaultValue (D 1)
        let activation =  activation |> Option.defaultValue Activation.None
        let init =  init |> Option.defaultValue (C.GlorotUniformInitializer())
        let pad  = pad |> Option.defaultValue false
        let bias = bias |> Option.defaultValue true
        let init_bias = init_bias |> Option.defaultValue 0.f
        let strides = strides |> Option.defaultValue (D 1)
        let name= name |> Option.defaultValue ""
       
        if dims filter_shape |> List.length > 2 then            
            failwith "ConvolutionTranspose2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)"

        let filter_shape = match filter_shape with D i -> Ds [i;i] | x -> x

        L.ConvolutionTranspose(
            convVar,
            filter_shape,
            num_filters,
            activation,
            init,
            pad,
            strides,
            sharing=true,
            bias=bias,
            init_bias=init_bias,
            name=name)
//
let x = () 
    C.ConvolutionTranspose(
def ConvolutionTranspose2D(filter_shape,        # 
                           num_filters,
                           activation=default_override_or(identity),
                           init=default_override_or(C.glorot_uniform()),
                           pad=default_override_or(False),
                           strides=1,
                           bias=default_override_or(True),
                           init_bias=default_override_or(0),
                           output_shape=None,
                           reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                           dilation=1,
                           name=''):
    '''
    ConvolutionTranspose2D(filter_shape, num_filters, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, output_shape=None, name='')

    Layer factory function to create a 2D convolution transpose layer with optional non-linearity.
    Same as `ConvolutionTranspose()` except that filter_shape is verified to be 2-dimensional.
    See `ConvolutionTranspose()` for extensive documentation.
    '''
    activation = get_default_override(ConvolutionTranspose2D, activation=activation)
    init       = get_default_override(ConvolutionTranspose2D, init=init)
    pad        = get_default_override(ConvolutionTranspose2D, pad=pad)
    bias       = get_default_override(ConvolutionTranspose2D, bias=bias)
    init_bias  = get_default_override(ConvolutionTranspose2D, init_bias=init_bias)
    output_shape = get_default_override(ConvolutionTranspose2D, output_shape=output_shape)
    if len(_as_tuple(filter_shape)) > 2:
         raise ValueError('ConvolutionTranspose2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)')
    filter_shape = _pad_to_shape((0,0), filter_shape, 'filter_shape')
    return ConvolutionTranspose(filter_shape, num_filters, activation, init, pad, strides, True, bias, init_bias, output_shape, reduction_rank=reduction_rank, dilation=dilation, name=name)
