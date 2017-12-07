// derived from this Python-based tutorial: 
// https://www.cntk.ai/pythondocs/CNTK_206A_Basic_GAN.html

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


let featureStreamName = "features"
let labelsStreamName = "labels"
let imageSize = 28 * 28
let numClasses = 10

let img_h, img_w = 28, 28
let kernel_h, kernel_w = 5, 5
let stride_h, stride_w = 2, 2

let g_input_dim = 100
let g_output_dim = img_h * img_w

let d_input_dim = g_output_dim
let isFast = true
    
// training config
let minibatch_size = 128u
let num_minibatches = if isFast then 5000 else 10000
let lr = 0.0002
let momentum = 0.5 //equivalent to beta1

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

// the strides to be of the same length along each data dimension
let gkernel,dkernel =
    if kernel_h = kernel_w then
        kernel_h,kernel_h
    else
        failwith "This tutorial needs square shaped kernel"

let gstride,dstride =
    if stride_h = stride_w then
       stride_h, stride_h
    else
        failwith "This tutorial needs same stride in all dims"

// Helper functions  
let bn_with_relu (x:Variable) =
    let h = L.BatchNormalization(x,map_rank=1)
    L.activation h Activation.ReLU

//use PReLU function to use a leak=0.2 since CNTK implementation
// of Leaky ReLU is fixed to 0.01
let bn_with_leaky_relu x leak =
    let h = L.BatchNormalization(x,map_rank=1)
    Activation.PReLU leak |> L.activation h 
    
let convolutional_generator (z:Variable)  =
    let defaultInit() = C.NormalInitializer(0.2)
    printfn "generator input shape: %A" z.Shape
    let s_h2, s_w2 = img_h / 2, img_w / 2 //Input shape (14,14)
    let s_h4, s_w4 = img_h / 4, img_w / 4 //Input shape (7,7)
    let gfc_dim = 1024
    let gf_dim = 64

    let h0 = L.Dense(z, D gfc_dim, init=defaultInit(), activation=Activation.NONE, name="h0")
    let h0 = !> (bn_with_relu !> h0)
    printfn "h0 shape: %A"  h0.Shape

    let h1 = L.Dense(h0, Ds [gf_dim *2; s_h4; s_w4], init=defaultInit(), activation=Activation.NONE, name= "h1")
    let h1 = !> (bn_with_relu !> h1)
    printfn "h1 shape: %A"  h1.Shape

    let h2 = L.ConvolutionTranspose2D(
                h1,
                D gkernel,
                num_filters=gf_dim*2,
                strides=D gstride,
                pad=true,
                output_shape=Ds[s_h2; s_w2],
                activation=Activation.NONE)
    let h2 = !> (bn_with_relu !>h2)
    printfn "h2 shape %A" h2.Shape

    let h3:Variable = 
        !> L.ConvolutionTranspose2D(
                h2,
                D gkernel,
                num_filters=1,
                strides=D gstride,
                pad=true,
                output_shape=Ds[img_h; img_w],
                activation=Activation.Sigmoid)
    printfn "h3 shape: %A" h3.Shape

    C.Reshape(h3, !- (D img_h * img_w))


let convolutional_discriminator (x:Variable)  =
    let dfc_dim = 1024
    let df_dim = 64
    printfn "Discriminator convolution shape %A" x.Shape
    let x = C.Reshape(x, !- (Ds [1; img_h; img_w]))

    let h0 = L.Convolution2D(!>x, D dkernel, num_filters=1, strides=D dstride)
    let h0 = !>(bn_with_leaky_relu !>h0 0.2)
    printfn "h0 shape : %A" h0.Shape 

    let h1 = L.Convolution2D(h0, D dkernel, num_filters=df_dim, strides=D dstride)
    let h1 = !>(bn_with_leaky_relu !>h1 0.2)
    printfn "h1 shape : %A" h1.Shape

    let h2 = L.Dense(h1, D dfc_dim, activation=Activation.NONE)
    let h2 = !>(bn_with_leaky_relu !>h2 0.2)
    printfn "h3 shape : %A" h2.Shape

    let h3 =  L.Dense(h2, D 1, activation=Activation.Sigmoid)
    printfn "h3 shape : %A" h3.Output.Shape

    h3


let build_graph noise_shape image_shape generator discriminator =
    let input_dynamic_axes = new AxisVector([|Axis.DefaultBatchAxis()|])
    let Z = C.InputVariable(noise_shape,dataType,input_dynamic_axes)
    let X_real = C.InputVariable(image_shape,dataType,input_dynamic_axes)
    let X_real_scaled = C.ElementDivide(X_real, scalar 255.0)

    //generator & discriminator 
    let X_fake:Function = generator Z
    let D_real:Function = discriminator !>X_real_scaled
    let D_fake = D_real.Clone(
                    ParameterCloningMethod.Share,
                    idict [X_real_scaled.Output, X_fake.Output])

    //loss functions
    //G_loss = 1.0 - C.log(D_fake)
    let G_loss = C.Minus(scalar 1.0, !> C.Log(!> D_fake))

    //D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))
    let D_loss = C.Negate(!> C.Plus(!> C.Log(!>D_real), !>C.Minus(scalar 1.0, !> D_fake))) 

    let G_learner = C.AdamLearner(
                        X_fake.Parameters() |> parmVector,
                        new TrainingParameterScheduleDouble(lr,1u),
                        new TrainingParameterScheduleDouble(momentum))

    let D_learner = C.AdamLearner(
                        D_real.Parameters() |> parmVector,
                        new TrainingParameterScheduleDouble(lr,1u),
                        new TrainingParameterScheduleDouble(momentum))


    let G_trainer = C.CreateTrainer(X_fake,G_loss,null,lrnVector [G_learner])
    let D_trainer = C.CreateTrainer(D_real,D_loss,null,lrnVector [D_learner])

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

let train (reader_train:MinibatchSource) generator discriminator =
    let X_real, X_fake, Z, G_trainer, D_trainer =
        build_graph 
            (shape [g_input_dim]) 
            (shape [d_input_dim])
            generator
            discriminator

    let featureStreamInfo = reader_train.StreamInfo(featureStreamName)
    let k =2 
    let print_frequency_mbsize = num_minibatches / 25
    let pp_G = p()
    let pp_D=  p()

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
        //hmmm python code does it twice

        if train_step % 100 = 0 then
            let l_D = D_trainer.PreviousMinibatchLossAverage()
            let l_G = G_trainer.PreviousMinibatchLossAverage()
            printfn "Minibatch: %d, D_loss=%f, G_loss=%f" train_step l_D l_G

    let G_trainer_loss = G_trainer.PreviousMinibatchLossAverage()
    Z, X_fake, G_trainer_loss
(*

let reader_train = minibatchSource

let G_input, G_output, G_trainer_loss = train reader_train 
                                              convolutional_generator 
                                              convolutional_discriminator

G_output.Save(Path.Combine(@"D:\repodata\fscntk","GeneratorDCGAN.bin"))

let noise = noise_sample 36
let outMap = idict[G_output.Output,(null:Value)]
G_output.Evaluate(idict[G_input,noise.data],outMap,device)
let imgs = outMap.[G_output.Output].GetDenseData<float32>(G_output.Output)

let sMin,sMax = Seq.collect (fun x->x) imgs |> Seq.min, Seq.collect (fun x->x) imgs |> Seq.max
let grays = 
    imgs
    //|> Seq.map (Seq.map (fun x-> if x < 0.f then 0uy else 255uy)>>Seq.toArray)
    |> Seq.map (Seq.map (fun x -> scaler (0.,255.) (float sMin, float sMax) (float x) |> byte) >> Seq.toArray)
    |> Seq.map (ImageUtils.toGray (28,28))
    |> Seq.toArray

ImageUtils.show grays.[0]
ImageUtils.showGrid (6,6) grays
*)

(*

//testing
let dx = new AxisVector([|Axis.DefaultDynamicAxis()|])
let x = new Variable(!-(Ds [NDShape.FreeDimension; 4]), VariableKind.Input, dataType, null, true, dx, false, "","1")
x.Shape
let w = new Parameter( !- (D NDShape.InferredDimension + D 10), dataType, 0.)
w.Shape
let t = C.Times(x,w, 1u)
t.Output.Shape
t.Output
let b = new Parameter( !- (D 10), dataType,0.,device,"b")
b.Shape
let m = C.Plus(!>t,b)
m.Output.Shape

*)