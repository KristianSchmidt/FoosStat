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
    let matchSummaryToTable (ms : MatchSummary) = 
        let header = Tags.THead [ TR [ TH [ Attr.ColSpan "3"; Attr.Class "text-center"; Text ms.StatName ] ]
                                  TR [ TH [ Text "" ]; TH [ Text "Red" ]; TH [ Text "Blue" ] ] ]
            
        let bodyElement name (redStat : Stat) (blueStat : Stat) =
            TR [ TD [ Text name ]; TD [ Text (sprintf "%O" redStat) ]; TD [ Text (sprintf "%O" blueStat) ] ]

        let setRow i (redStat : Stat) (blueStat : Stat) = bodyElement (sprintf "Set %i" (i + 1)) redStat blueStat

        let totalRow =
            bodyElement "Total" ms.Red.MatchTotal ms.Blue.MatchTotal
            -< [ Attr.Class "active" ]

        let body = Tags.TBody ((ms.Red.SetTotals,ms.Blue.SetTotals) ||> List.mapi2 setRow) -< [ totalRow ]

        Table [ Attr.Class "table table-striped table-hover" ] -< [header; body]

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
            Div [ Attr.Class "container" ] -< (chunks |> Seq.map makeRow)
            |> textDiv.Append

        let mainDiv =
            Div [
                Attr.Class "main-element"
            ]

        addTable ()
        
        Div [ mainDiv
              textDiv ] 