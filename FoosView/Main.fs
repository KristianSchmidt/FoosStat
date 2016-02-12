namespace FoosView

open WebSharper.Html.Server
open WebSharper
open WebSharper.Sitelets
open WebSharper.Resources

type EndPoint =
    | [<EndPoint "GET /">] Home

module Templating =
    open System.Web

    type Page =
        {
            Title : string
            Body : list<Element>
        }

    let MainTemplate =
        Content.Template<Page>("~/Main.html")
            .With("title", fun x -> x.Title)
            .With("body", fun x -> x.Body)
            
    let Main ctx endpoint title body : Async<Content<EndPoint>> =
        Content.WithTemplate MainTemplate
            {
                Title = title
                Body = body
            }

module Site =

    let HomePage ctx =
        Templating.Main ctx EndPoint.Home "FoosStat" [
            Div [ClientSide <@ Client.Main() @>]
        ]

    [<Website>]
    let Main =
        Application.MultiPage (fun ctx action ->
            match action with
            | Home -> HomePage ctx
        )

[<Sealed>]
type Website() =
    interface IWebsite<EndPoint> with
        member this.Sitelet = Site.Main
        member this.Actions = [Home]

type StyleResource() =
    inherit BaseResource("bootstrap.css")

[<assembly: Website(typeof<Website>)>]
[<assembly: Require(typeof<StyleResource>)>]
do ()
