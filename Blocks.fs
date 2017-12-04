namespace CNTKWrapper
open CNTK
open System
type C = CNTKLib

//based on python blocks module (see CNTK Python API for documentation)

module Blocks =

    type B =

        static member _initializer_with_rank (init, ?output_rank, ?filter_rank) =
            let output_rank = output_rank |> Option.defaultValue C.SentinelValueForInferParamInitRank
            let filter_rank = filter_rank |> Option.defaultValue C.SentinelValueForInferParamInitRank
            C.RandomInitializerWithRank(init, output_rank, filter_rank)

