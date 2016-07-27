namespace App

open Fable.Core
open Fuse
open Fable.Import
open FoosStat
open Domain

module StatsPage =
    
    let matchSummary = Observable.createList [||]
    
    let updateSummary (obs : Observable.IObservable<_>) =
        if (allBalls.length > 0) then
            let set = obs.toArray() |> Array.map List.ofArray |> List.ofArray |> List.map Ball |> Set

            let m = { Sets = [ set ]; Red = SingleTeam "Red"; Blue = SingleTeam "Blue" }

            let summ = App.Analytics.matchSummary m |> List.toArray
            matchSummary.replaceAll summ
    
    allBalls.addSubscriber updateSummary
    