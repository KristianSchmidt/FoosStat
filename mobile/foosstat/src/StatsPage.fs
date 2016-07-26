namespace App

open Fable.Core
open Fuse
open Fable.Import
open FoosStat
open Domain

module StatsPage =
    let matchSummary = Observable.createList [||]

    let updateSummary (obs : Observable.IObservable<_>) =
        let set = obs.toArray() |> Array.map List.ofArray |> List.ofArray |> List.map Ball |> Set

        let m = { Sets = [ set ]; Red = SingleTeam "Red"; Blue = SingleTeam "Blue" }

        let summ = App.Analytics.matchSummary m |> List.toArray
        matchSummary.replaceAll summ

    allBalls.addSubscriber updateSummary
    
    let currScore = FoosStat.currScore
    let currBall = FoosStat.currBall

    let restartFromGoal goal fiveBar =
        currBall.add goal
        let arr = currBall.toArray()
        allBalls.add arr
        currBall.clear()
        currBall.add fiveBar

    let addGoalBlue _ = restartFromGoal (Goal Blue) (Possession (Red, Midfield))
    let addGoalRed  _ = restartFromGoal (Goal Red)  (Possession (Blue, Midfield))

    type Testing = { testName : string }

    let testers = Observable.createList [| {testName = "Test 1"}; { testName = "Test 2"}|]

    let addTester _ = testers.add {testName = "Test X"}