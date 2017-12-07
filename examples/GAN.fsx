// derived from this Python-based tutorial: 
// https://www.cntk.ai/pythondocs/CNTK_206B_DCGAN.html

#load "../CNTK.fsx"
#load "../Probability.fs"
//#load "MNIST-CNN.fsx"
#load "../FsBase.fs"
#load "../Blocks.fs"
#load "../Layers.fs"
#load "../ImageUtils.fs"

open System
open System.IO
open System.Collections.Generic

open CNTK
open CNTKWrapper.FsBase
open CNTKWrapper.Blocks
open CNTKWrapper.Layers
type C = CNTKLib
open ImageUtils

let g_input_dim = 100
let g_hidden_dim = 128
let g_output_dim = 784
let d_input_dim = g_output_dim
let d_hidden_dim = 128
let d_output_dim = 1
let isFast = false
let minibatch_size = 1024u
let num_minibatches = if isFast then 300 else 40000
let lr = 0.00005

let featureStreamName = "features"
let labelsStreamName = "labels"
let imageSize = 28 * 28
let numClasses = 10

let streamConfigurations = 
    ResizeArray<StreamConfiguration>(
        [
            new StreamConfiguration(featureStreamName, imageSize)    
            new StreamConfiguration(labelsStreamName, numClasses)
        ]
        )

let cntk_samples_folder = @"C:\s\cntk\Examples\Image\DataSets\MNIST" //from CNTK download

let minibatchSource = 
    MinibatchSource.TextFormatMinibatchSource(
        Path.Combine(cntk_samples_folder, "Train-28x28_cntk_text.txt"), 
        streamConfigurations, 
        MinibatchSource.InfinitelyRepeat)

let uniform_sample size =
    [|
        for _ in 1 .. size do
            let r = Probability.RNG.Value.NextDouble()
            yield  (float32 r - 0.5f) * 2.0f  //[-1,+1]
    |] 
    //uniform_sample 20

let noise_sample num_samples =
    let vals = uniform_sample  (num_samples * g_input_dim)
    let inp = Value.CreateBatch(shape [g_input_dim], vals, device)
    new MinibatchData(inp,uint32 minibatch_size)

let generator z =
    let h1 = L.Dense(z,D g_hidden_dim,activation=Activation.ReLU,name="h1")
    L.Dense(new Variable(h1),D g_output_dim,activation=Activation.Tanh,name="outG")

let discriminator x =
    let h1 = L.Dense(x,D d_hidden_dim,activation=Activation.ReLU,name="h1")
    L.Dense(new Variable(h1),D d_output_dim,activation=Activation.Sigmoid,name="outD")

let dt = DataType.Float //default data type
let scalar x = Constant.Scalar(dt,x)


let build_graph noise_shape image_shape g_progress_printer d_progress_printer =
    let input_dynamic_axes = new AxisVector([|Axis.DefaultBatchAxis()|])
    let Z = C.InputVariable(noise_shape,dt,input_dynamic_axes)
    let X_real = C.InputVariable(image_shape,dt,input_dynamic_axes)
    let X_real_scaled = C.Minus(!> C.ElementTimes(X_real, scalar (2.0/255.0)),scalar 1.0)

    //generator & discriminator 
    let X_fake = generator Z
    let D_real = discriminator !> X_real_scaled
    let D_fake = D_real.Clone(
                    ParameterCloningMethod.Share,
                    idict [X_real_scaled.Output, X_fake.Output])

    //loss functions
    //G_loss = 1.0 - C.log(D_fake)
    let G_loss = C.Minus(scalar 1.0, !> C.Log(!> D_fake))

    //D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))
    let D_loss = C.Negate(!> C.Plus(!> C.Log(!>D_real), !> C.Minus(scalar 1.0, !> D_fake))) 

    let G_learner = C.FSAdaGradLearner(
                        X_fake.Parameters() |> parmVector,
                        new TrainingParameterScheduleDouble(lr,1u),
                        new TrainingParameterScheduleDouble(0.9985724484938566))

    let D_learner = C.FSAdaGradLearner(
                        D_real.Parameters() |> parmVector,
                        new TrainingParameterScheduleDouble(lr,1u),
                        new TrainingParameterScheduleDouble(0.9985724484938566))


    let G_trainer = C.CreateTrainer(X_fake,G_loss,null,lrnVector [G_learner], g_progress_printer)
    let D_trainer = C.CreateTrainer(D_real,D_loss,null,lrnVector [D_learner], d_progress_printer)

    X_real, X_fake, Z, G_trainer, D_trainer   
    
//not sure how to use ProgressWriter - don't see output of this code
let p() = {new ProgressWriter(1u,1u,1u,1u,1u,1u) with
            override x.OnWriteTrainingUpdate(s1,s2,s3,s4) =
                printfn "%A, loss=%A * %A, metric = %A" s2 s2 s3 s4
            override x.OnWriteTrainingSummary(s1,s2,s3,s4, s5, s6) =
                printfn "%A, loss=%A * %A, metric = %A" s2 s2 s3 s4
            override x.Write(s1,s2) = 
                printfn "%s=%f" s1 s2
         }

let train (reader_train:MinibatchSource) =
    let featureStreamInfo = reader_train.StreamInfo(featureStreamName)
    let k =2 
    let print_frequency_mbsize = num_minibatches / 50
    let pp_G = p()
    let pp_D=  p()
    let X_real, X_fake, Z, G_trainer, D_trainer =
        build_graph 
            (shape [g_input_dim]) 
            (shape [d_input_dim])
            (prgwVector [pp_G])
            (prgwVector [pp_D])

    for train_step in 1 .. num_minibatches do

        //train the discriminator for k steps
        for gen_train_step in 1..k do
            let Z_data = noise_sample (int minibatch_size)
            let X_data = reader_train.GetNextMinibatch(minibatch_size)
            if X_data.[featureStreamInfo].numberOfSamples = Z_data.numberOfSamples then
                let batch_inputs = 
                    idict
                        [
                            X_real, X_data.[featureStreamInfo]
                            Z     , Z_data
                        ]
                D_trainer.TrainMinibatch(batch_inputs,device) |> ignore

        //train generator
        let Z_data = noise_sample (int minibatch_size)
        let batch_inputs = idict [Z, Z_data]
        let b = G_trainer.TrainMinibatch(batch_inputs,device) //|> ignore
        if train_step % 100 = 0 then
            let l_D = D_trainer.PreviousMinibatchLossAverage()
            let l_G = G_trainer.PreviousMinibatchLossAverage()
            printfn "Minibatch: %d, D_loss=%f, G_loss=%f" train_step l_D l_G

    let G_trainer_loss = G_trainer.PreviousMinibatchLossAverage()
    Z, X_fake, G_trainer_loss

(*
*)

let reader_train = minibatchSource
let G_input, G_output, G_trainer_loss = train reader_train
G_output.Save(Path.Combine(@"D:\repodata\fscntk","Generator.bin"))

let noise = noise_sample 36
let outMap = idict[G_output.Output,(null:Value)]
G_output.Evaluate(idict[G_input,noise.data],outMap,device)
let imgs = outMap.[G_output.Output].GetDenseData<float32>(G_output.Output)

let sMin,sMax = Seq.collect (fun x->x) imgs |> Seq.min, Seq.collect (fun x->x) imgs |> Seq.max
let grays = 
    imgs
    |> Seq.map (Seq.map (fun x-> if x < 0.f then 0uy else 255uy)>>Seq.toArray)
    //|> Seq.map (Seq.map (fun x -> scaler (0.,255.) (float sMin, float sMax) (float x) |> byte) >> Seq.toArray)
    |> Seq.map (ImageUtils.toGray (28,28))
    |> Seq.toArray

ImageUtils.show grays.[0]
ImageUtils.showGrid (6,6) grays

