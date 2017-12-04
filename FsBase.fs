namespace CNTKWrapper

// F# specific supporting and utility functions 

module FsBase =

    //utility operator for F# implicit conversions 
    let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

    let yourself x = x
