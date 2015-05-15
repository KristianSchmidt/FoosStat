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

        let table = Table [ Tags.THead [ TR [ TH [ Attr.ColSpan "3"; Attr.Align "center" ] -< [ TD [ Text "Goals" ] ] ]
                                         TR [ TH [ Text "" ]; TH [ Text "Red" ]; TH [ Text "Blue" ] ] ]
                            Tags.TBody [ TR [ TD [ Text "Set 1" ]; TD [ Text "Set 1" ]; TD [ Text "Set 1" ] ] ] ]

        let matchSummaryToTable (MatchSummary(name,
                                              (col1,PlayerSummary(matchTotal1,setTotals1)),
                                              (col2,PlayerSummary(matchTotal2,setTotals2))
                                )) =
            let header = Tags.THead [ TR [ TH [ Attr.ColSpan "3"; Attr.Align "center" ] -< [ TD [ Text name ] ] ]
                                      TR [ TH [ Text "" ]; TH [ Text "Red" ]; TH [ Text "Blue" ] ] ]
            let bodyElement i (redStat : Stat) (blueStat : Stat) =
                TR [ TD [ Text (sprintf "Set %i" i) ]; TD [ Text (sprintf "%O" redStat) ]; TD [ Text (sprintf "%O" blueStat) ] ]

            let body = Tags.TBody ((setTotals1,setTotals2) ||> List.mapi2 bodyElement)

            Table [ header; body ]
        
        let printMatchSummary (MatchSummary(name,
                                (col1,PlayerSummary(matchTotal1,setTotals1)),
                                (col2,PlayerSummary(matchTotal2,setTotals2))
                                )) =
            let newLineConcat s1 s2 = s1 + "\r\n" + s2
            let header = sprintf "%s" name + "\r\n" + (sprintf "%A\t-\t%A" col1 col2)
            let content = (setTotals1,setTotals2) ||> List.zip |> List.map (fun (s1,s2) -> sprintf "%O\t-\t%O" s1 s2) |> List.reduce newLineConcat
            let total = sprintf "Match total: %O\t-\t%O" matchTotal1 matchTotal2
            header + "\r\n" + content + "\r\n" + total

        let parseGame text =
            Parser.parseGame text |> Seq.map Parser.parseSet |> List.ofSeq |> Match

        let input = 
            Controls.ReadOnlyTextArea game
            |> Enhance.WithSubmitButton
        let label = Text "Rod"
        let textDiv = Div []
        let appendSummary (text : string) =
            textDiv.Clear()
            let summary = parseGame text |> matchSummary
            textDiv.Append (matchSummaryToTable (List.head summary))
        let mainDiv =
            Div [
                Attr.Class "main-element"
                label
                input.Run (fun text -> appendSummary text)
            ]
        Div [ mainDiv
              textDiv ] 