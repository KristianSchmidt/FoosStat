namespace App

open Fable.Core
open Fuse
open Fable.Import
open FoosStat
open Domain

module GamePage =
    let currScore = FoosStat.currScore
    let currBall = FoosStat.currBall

    let addBlue2 _ = currBall.add (Possession (Blue, Defence))
    let addBlue5 _ = currBall.add (Possession (Blue, Midfield))
    let addBlue3 _ = currBall.add (Possession (Blue, Attack))
    let addRed2  _ = currBall.add (Possession (Red, Defence))
    let addRed5  _ = currBall.add (Possession (Red, Midfield))
    let addRed3  _ = currBall.add (Possession (Red, Attack))

    let restartFromGoal goal fiveBar =
        currBall.add goal
        let arr = currBall.toArray()
        allBalls.add arr
        currBall.clear()
        currBall.add fiveBar

    let addGoalBlue _ =
        restartFromGoal (Goal Blue) (Possession (Red, Midfield))
    let addGoalRed  _ = restartFromGoal (Goal Red)  (Possession (Blue, Midfield))

    let clearBall _ = currBall.clear()
    let resetGame _ =
        currBall.clear()
        allBalls.clear()

    let currBallText = FoosStat.currBallText