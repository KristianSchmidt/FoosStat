namespace FoosStatView

open IntelliFactory.WebSharper
open IntelliFactory.WebSharper.Formlets
open IntelliFactory.WebSharper.JavaScript
open IntelliFactory.WebSharper.Html.Client
open Domain
open Parser
open Analytics

[<JavaScript>]
module Client =

    [<Literal>]
    let game = @"set 1
b5,b5,b3,g_b
r5,r3,r2,b2,g_b
r5,r3,g_r
b5,b2,b5,r5,r3,b2,b3,b2,r2,r2,b2,r2,r3,g_r
b5,b2,b3,r2,b3,g_b
r5,r3,b2,b2,g_b
r5,r3,g_r
b5,r5,b5,g_b
set 2
r5,r3,b2,r5,r3,b2,b2,r2,g_r
b5,b3,r2,b2,r2,r3,r3,g_r
b5,b5,b3,g_b
r5,b5,r2,r3,g_r
b5,b3,g_b
r5,r3,b2,r5,b5,r5,r3,b2,r2,b2,b2,r2,b2,b3,g_b
r5,r5,r3,g_r
b5,b3,g_b
r5,r3,r3,r5,r3,r2,r2,r2,b2,b5,b3,r2,r2,r2,r2,b2,g_b
set 3
r5,r3,g_r
b5,b5,r5,r3,g_r
b5,r5,b2,r3,r5,r3,r3,g_r
b5,b5,b3,r2,r3,b5,b5,b5,b3,r2,r5,r3,b2,r2,b5,r5,r3,b2,r3,b2,r3,b2,b5,b3,g_b
r5,r3,r2,b2,b5,b5,b3,g_b
r5,r3,g_r
b5,b5,b3,r5,b5,r5,r3,b5,b2,b3,r2,r3,g_r
set 4
b5,r2,r3,g_r
b5,b5,r2,b2,b2,r3,g_r
b5,r5,r3,g_r
b5,r3,g_r
b5,r5,r5,r3,r3,r3,b2,r5,r3,g_r
set 5
b5,b5,b3,r2,b2,g_b
r5,r5,b5,r3,r3,g_r
b5,r2,r3,g_r
b5,r5,b2,r2,b2,r2,r3,r3,g_r
b5,b3,r2,r5,b5,r2,r2,b5,b3,b5,b5,r5,b2,g_b
r5,b5,b2,r5,r3,b2,r5,b5,r5,r3,g_r
b5,b5,g_b
r5,b5,b5,r5,r3,b5,b5,b5,b3,g_b
r5,r3,r3,g_r
b5,r2,b5,r2,r5,r3,r3,r3,g_r"
    
    let Main () =
        (*
        let input = Input [Text ""]
        let label = Div [Text ""]
        let test = Attack
        
        let selform = ["Attack", Attack; "Defence", Defence]
                      |> Controls.Select 0
        
        Div [
            label
            selform
        ]*)

        let parseGame text =
            Parser.parseGame text |> Seq.map Parser.parseSet |> List.ofSeq |> Match

        let input = 
            Controls.ReadOnlyTextArea game
            |> Enhance.WithSubmitButton
        let label = Text "Rod"
        Div [
            Attr.Class "main-element"
            label
            input.Run (fun text -> JS.Alert(sprintf "%A" (parseGame text)))
        ]
        (*
        let summaries = matchSummary foosmatch

        let printMatchSummary (MatchSummary(name,
                                (col1,PlayerSummary(matchTotal1,setTotals1)),
                                (col2,PlayerSummary(matchTotal2,setTotals2))
                                )) =
            printfn "%s" name
            printfn "%A\t-\t%A" col1 col2
            (setTotals1,setTotals2) ||> List.zip |> List.iter (fun (s1,s2) -> printfn "%O\t-\t%O" s1 s2)
            printfn "Match total: %O\t-\t%O" matchTotal1 matchTotal2
            printfn ""

        summaries |> List.iter printMatchSummary *)