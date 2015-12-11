namespace FoosStatView

open IntelliFactory.WebSharper.Html.Server
open IntelliFactory.WebSharper
open IntelliFactory.WebSharper.Sitelets
open IntelliFactory.WebSharper.Resources

type Action =
    | Home

module Controls =

    [<Sealed>]
    type EntryPoint() =
        inherit Web.Control()

        [<JavaScript>]
        override __.Body =
            Client.Main() :> _

module Skin =
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

    let WithTemplate title body : Content<Action> =
        Content.WithTemplate MainTemplate <| fun context ->
            {
                Title = title
                Body = body context
            }

module Site =

    let ( => ) text url =
        A [HRef url] -< [Text text]

    let HomePage =
        Skin.WithTemplate "FoosStat" <| fun ctx ->
            [
                Div [new Controls.EntryPoint()]
            ]

    let Main =
        Sitelet.Sum [
            Sitelet.Content "/" Home HomePage
        ]

[<Sealed>]
type Website() =
    interface IWebsite<Action> with
        member this.Sitelet = Site.Main
        member this.Actions = [Home]

type StyleResource() =
    inherit BaseResource("bootstrap.css")

[<assembly: Website(typeof<Website>)>]
[<assembly: Require(typeof<StyleResource>)>]
do ()
