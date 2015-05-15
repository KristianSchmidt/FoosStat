namespace FoosStatView

open IntelliFactory.WebSharper
open IntelliFactory.WebSharper.Formlets
open IntelliFactory.WebSharper.JavaScript
open IntelliFactory.WebSharper.Html.Client
open Domain
open Parser
open Analytics
open Games

[<JavaScript>]
module Client = 
    let matchSummaryToTable (MatchSummary(name,
                                              (col1,PlayerSummary(matchTotal1,setTotals1)),
                                              (col2,PlayerSummary(matchTotal2,setTotals2))
                                )) =
            let header = Tags.THead [ TR [ TH [ Attr.ColSpan "3"; Attr.Align "center" ] -< [ TD [ Text name ] ] ]
                                      TR [ TH [ Text "" ]; TH [ Text "Red" ]; TH [ Text "Blue" ] ] ]
            let bodyElement i (redStat : Stat) (blueStat : Stat) =
                TR [ TD [ Text (sprintf "Set %i" (i + 1)) ]; TD [ Text (sprintf "%O" redStat) ]; TD [ Text (sprintf "%O" blueStat) ] ]

            let body = Tags.TBody ((setTotals1,setTotals2) ||> List.mapi2 bodyElement)

            Table [ header; body ]

    let Main () =
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
        
        let textDiv = Div []
        let appendSummary (text : string) =
            textDiv.Clear()
            let summary = parseGame text |> matchSummary
            textDiv.Append (matchSummaryToTable (List.head summary))
        let mainDiv =
            Div [
                Attr.Class "main-element"
                input.Run (fun text -> appendSummary text)
            ]
        Div [ mainDiv
              textDiv ] 