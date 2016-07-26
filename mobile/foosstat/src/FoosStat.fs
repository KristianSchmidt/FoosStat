namespace App
open Fable.Core
open Fuse
open Fable.Import
open Fable.Import.Fetch
open Domain

module FoosStat =
    let private eventToString = function
                                | Goal(Blue) -> "GB"
                                | Goal(Red)  -> "GR"
                                | Possession(Red, Defence) -> "R2"
                                | Possession(Red, Midfield) -> "R5"
                                | Possession(Red, Attack) -> "R3"
                                | Possession(Blue, Defence) -> "B2"
                                | Possession(Blue, Midfield) -> "B5"
                                | Possession(Blue, Attack) -> "B3"
                                
    
    let allBalls = Observable.createList [||]

    let set = allBalls.toArray() |> Array.map List.ofArray |> List.ofArray |> List.map Ball |> Set

    let m = { Sets = [ set ]; Red = SingleTeam "Red"; Blue = SingleTeam "Blue" }

    let currBall : Observable.IObservable<Event> = Observable.createList [||]

    let private currBallToString () = currBall.toArray () |> Array.map eventToString |> String.concat " - "

    let currBallText = Observable.createWith (currBallToString())
    
    currBall.addSubscriber (fun _ -> currBallText.value <- currBallToString ())
    
    let private getCurrScoreString () =
        let balls = allBalls.toArray() 
        let goals = balls |> Array.map Array.last
        
        let redGoals  = goals |> Array.filter ((=) (Goal Red))  |> Array.length
        let blueGoals = goals |> Array.filter ((=) (Goal Blue)) |> Array.length
        sprintf "RED %i - BLUE %i" redGoals blueGoals

    let currScore = Observable.createWith (getCurrScoreString ())
    allBalls.addSubscriber (fun _ -> currScore.value <- getCurrScoreString ())
