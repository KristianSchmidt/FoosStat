namespace App

open Fable.Core
open Fuse
open Fable.Import
open FoosStat
open Domain

module StatsPage =
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