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
            
            Table [ Attr.Class "table table-striped table-hover " ] -< [header; body]

    let Main () =
        let parseGame text =
            Parser.parseGame text |> Seq.map Parser.parseSet |> List.ofSeq |> Match

        let textDiv = Div []
        
        let makeRow summaryChunk = 
            let wrap ms = Div [ Attr.Class "col-md-6" ] -< [ matchSummaryToTable ms ]

            Div [ Attr.Class "row" ] -< (summaryChunk |> Seq.map wrap)
        
        let windowChunk n (xs : MatchSummary seq) =
            let range = [0 .. Seq.length xs]
            Seq.windowed n xs 
            |> Seq.zip range
            |> Seq.filter (fun d -> (fst d) % n = 0)
            |> Seq.map(fun x -> (snd x))

        let addTable () =
            textDiv.Clear()
            let summary = parseGame game |> matchSummary
            let chunks = summary |> windowChunk 2
            chunks |> Seq.map makeRow
            |> Div
            |> textDiv.Append

        let mainDiv =
            Div [
                Attr.Class "main-element"
            ]

        addTable ()
        
        Div [ mainDiv
              textDiv ] 