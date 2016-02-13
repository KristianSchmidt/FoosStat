namespace Analytics

open WebSharper
open WebSharper.JavaScript
open Domain

[<JavaScript>]
[<AutoOpen>]
module Analytics =
    let sumStats (f : Ball -> Stat list) game =
        let statPerSet (Set(balls)) = balls |> List.collect f |> Stat.sum
        let setStats = game.Sets |> List.map statPerSet
        let matchTotal = setStats |> Stat.sum
        { MatchTotal = matchTotal; SetTotals = setStats }

    let generateMatchSummary game (MatchStat(name,f)) =
        let redf = f Red
        let bluef = f Blue
        let redSummary = sumStats redf game
        let blueSummary = sumStats bluef game
        { StatName = name; RedTeam = game.Red; BlueTeam = game.Blue; Red = redSummary; Blue = blueSummary }

    let pairwiseStat f (ball : Ball) = ball.pairwise |> Seq.map f |> List.ofSeq

    let goals =
        let goals' color (ball : Ball) =
            let goalStat : Event * Event -> Stat = function
                                                   | _,Goal(c) when c = color -> NumberStat(1)
                                                   | _ -> NumberStat(0)
            ball |> pairwiseStat goalStat
        MatchStat("Goals", goals')

    let genericNumber initial success (e1,e2) =
        match (initial e1),(success e2) with
        | true,true -> NumberStat(1)
        | _ -> NumberStat(0)

    let genericTryFail initial success (e1,e2) =
        match (initial e1),(success e2) with
        | true,true  -> TryFailStat(1,1)
        | true,false -> TryFailStat(0,1)
        | _          -> TryFailStat(0,0)

    let genericTryFailRecatch initial success (e1,e2) =
        match (initial e1),(initial e2),(success e2) with
        | true,_,true      -> TryFailStat(1,1) // A try and a success
        | true,true,_      -> TryFailStat(0,0) // We are at the same initial rod for both events, doesn't count
        | true,false,false -> TryFailStat(0,1) // We fail and don't get it on the same rod
        | _                -> TryFailStat(0,0)

    let threeBarInit color = function | Possession(c,Attack) when c = color -> true | _ -> false
    let threeBarSucc color = function | Goal(c)              when c = color -> true | _ -> false

    let midfieldInit color = function | Possession(c, Midfield) when c = color -> true | _ -> false
    let midfieldSucc color = function | Possession(c, Attack)   when c = color -> true | _ -> false

    let twoBarGoalInit color = function | Possession(c, Defence) when c = color -> true | _ -> false
    let twoBarGoalSucc color = function | Goal(c) when c = color -> true | _ -> false

    let matchStat name init succ tryfail =
        let calcFunc color ball =
            let stat = tryfail (init color) (succ color)
            ball |> pairwiseStat stat
        MatchStat(name, calcFunc)

    let threeBarGoals        = matchStat "Three bar goals/shots" threeBarInit threeBarSucc genericTryFail
    let threeBarGoalsRecatch = matchStat "Three bar goals/poss"  threeBarInit threeBarSucc genericTryFailRecatch
    let fiveBarPasses        = matchStat "Five bar passes/atts"  midfieldInit midfieldSucc genericTryFail
    let fiveBarPassesRecatch = matchStat "Five bar passes/poss"  midfieldInit midfieldSucc genericTryFailRecatch
    let twoBarGoals          = matchStat "Two bar goals"         twoBarGoalInit twoBarGoalSucc genericNumber

    let gameStats = [goals; twoBarGoals; threeBarGoals; threeBarGoalsRecatch; fiveBarPasses; fiveBarPassesRecatch]

    let matchSummary game = gameStats |> List.map (generateMatchSummary game)