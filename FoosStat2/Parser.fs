namespace Parser

open WebSharper
open WebSharper.JavaScript
open Domain
open System
open System.Text.RegularExpressions

[<AutoOpen>]
[<JavaScript>]
module Parser = 
    let parseEvent = function
        | "r2" -> Possession(Red, Defence)
        | "r5" -> Possession(Red, Midfield)
        | "r3" -> Possession(Red, Attack)
        | "b2" -> Possession(Blue, Defence)
        | "b5" -> Possession(Blue, Midfield)
        | "b3" -> Possession(Blue, Attack)
        | "g_r" -> Goal(Red)
        | "g_b" -> Goal(Blue)
        | x -> failwith (sprintf "Non specified event encountered: \"%s\"" x)

    let parseBall (ballLine : string) = 
        let eventStrings = ballLine.Split [|','|]
        let events = eventStrings |> Array.map parseEvent
        Ball(List.ofArray events)

    let parseSet (setLines : seq<string>) = Set(List.ofSeq (Seq.map parseBall setLines))

    let parseGame (text : string) =
        let lines = text.Split([|"\r\n"|], StringSplitOptions.RemoveEmptyEntries)
        let isSetLine (lineNumber : int) (line : string) =
            match line.StartsWith("set") with
            | true -> Some(lineNumber)
            | false -> None

        let parseName prefix =
            lines
            |> Array.tryFind (fun s -> s.StartsWith prefix)
            |> Option.map (fun s -> s.Replace(prefix,"").Split([|'/'|]))
            |> Option.bind (fun arr -> if (arr.Length = 1) then
                                            SingleTeam arr.[0] |> Some
                                       else if (arr.Length = 2) then
                                            DoubleTeam(arr.[0],arr.[1]) |> Some
                                       else
                                            None
                           )

        let indicesBase = Array.mapi isSetLine lines |> Array.choose id
        let indices = Array.append indicesBase [| lines.Length |]
        let intervals = seq { for i = 0 to indices.Length - 2 do yield (indices.[i],indices.[i+1]) }
        let sets =            
            intervals
            |> Seq.map (fun (a,b) -> seq { for i = a+1 to b-1 do yield lines.[i] })
            |> Seq.map parseSet
            |> List.ofSeq

        let blue = defaultArg (parseName "BLUE:") (SingleTeam "Blue")
        let red  = defaultArg (parseName "RED:") (SingleTeam "Red")
        { Sets = sets; Red = red; Blue = blue }
