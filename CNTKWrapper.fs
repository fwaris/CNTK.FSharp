module CNTKWrapper
open CNTK
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

let prgwVector (pws:ProgressWriter seq) =
    let pwv = new ProgressWriterVector(Seq.length pws)
    pws |>  Seq.iter pwv.Add
    pwv

type Activation = 
    | None
    | ReLU
    | Sigmoid
    | Tanh
    | LeakyReLU

let FullyConnectedLinearLayer(
    input:Variable, 
    outputDim:int, 
    device:DeviceDescriptor,
    outputName:string) : Function =

    let inputDim = input.Shape.[0]

    let timesParam = 
        new Parameter(
            shape [outputDim; inputDim], 
            DataType.Float,
            C.GlorotUniformInitializer(
                float C.DefaultParamInitScale,
                C.SentinelValueForInferParamInitRank,
                C.SentinelValueForInferParamInitRank, 
                uint32 1),
            device, 
            "timesParam")

    let timesFunction = 
        new Variable(C.Times(timesParam, input, "times"))

    let plusParam = new Parameter(shape [ outputDim ], 0.0f, device, "plusParam")
    C.Plus(plusParam, timesFunction, outputName)

let Dense(
    input:Variable, 
    outputDim:int,
    device:DeviceDescriptor,
    activation:Activation, 
    outputName:string) : Function =

    let input : Variable =
        if (input.Shape.Rank <> 1)
        then
            let newDim = input.Shape.Dimensions |> Seq.reduce(fun d1 d2 -> d1 * d2)
            new Variable(C.Reshape(input, shape [ newDim ]))
        else input

    let fullyConnected : Function = 
        FullyConnectedLinearLayer(input, outputDim, device, outputName)
    
    match activation with
    | Activation.None       -> fullyConnected
    | Activation.ReLU       -> C.ReLU       !>fullyConnected
    | Activation.LeakyReLU  -> C.LeakyReLU  !>fullyConnected
    | Activation.Sigmoid    -> C.Sigmoid    !>fullyConnected
    | Activation.Tanh       -> C.Tanh       !>fullyConnected
