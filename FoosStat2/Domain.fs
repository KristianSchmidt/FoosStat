﻿namespace Domain

open WebSharper
open WebSharper.JavaScript


[<JavaScript>]
[<AutoOpen>]
module Domain =
    type Team =
        | SingleTeam of string
        | DoubleTeam of string * string

        override this.ToString() = match this with
                                   | SingleTeam(s) -> s
                                   | DoubleTeam(s1,s2) -> sprintf "%s/%s" s1 s2

    type PlayerColor = 
        | Red
        | Blue
    
        member color.otherColor =
            match color with | Red -> Blue | Blue -> Red

    type Rod =
        | Defence
        | Midfield
        | Attack

    type Event =
        | Possession of PlayerColor * Rod
        | Goal of PlayerColor

    let (|SameColor|_|) color =
        function
        | Possession(c,_),Possession(c',_) when c = color && c' = color -> Some(SameColor)
        | Possession(c,_),Goal(c') when c = color && c' = color -> Some(SameColor)
        | Goal(c),Possession(c',_) when c = color && c' = color -> Some(SameColor)
        | Goal(c),Goal(c') when c = color && c' = color -> Some(SameColor)
        | _ -> None

    type Ball = 
        | Ball of Event list

        with member this.pairwise = match this with
                                    | Ball(events) -> events |> Seq.pairwise

    type Set =
        | Set of Ball list

    type Match = { Sets : Set list; Red : Team; Blue : Team }

    type Stat =
        | NumberStat of num : int
        | TryFailStat of successes : int * attempts : int

        override this.ToString() =
            match this with
            | NumberStat(num) -> sprintf "%i" num
            | TryFailStat(0,0) -> sprintf "0 / 0 (N/A)"
            | TryFailStat(suc,att) -> sprintf "%i / %i (%.0f %%)" suc att (100.0 * (float)suc / (float)att)

        static member (+) (s1, s2) =
            match (s1,s2) with
            | NumberStat(n1),NumberStat(n2) -> NumberStat(n1+n2)
            | TryFailStat(suc1,att1),TryFailStat(suc2,att2) -> TryFailStat(suc1+suc2,att1+att2)
            | _ -> failwith "Type mismatch: Cannot sum different types of stats"

        static member sum (stats : Stat seq) = stats |> Seq.reduce (fun x y -> match (x,y) with
                                                                               | NumberStat(n1),NumberStat(n2) -> NumberStat(n1+n2)
                                                                               | TryFailStat(suc1,att1),TryFailStat(suc2,att2) -> TryFailStat(suc1+suc2,att1+att2)
                                                                               | _ -> failwith "Type mismatch: Cannot sum different types of stats")

    type PlayerSummary = { MatchTotal : Stat; SetTotals : Stat list }

    type MatchSummary = { StatName : string; RedTeam : Team; BlueTeam : Team; Red : PlayerSummary; Blue : PlayerSummary }

    type MatchStat = | MatchStat of name : string * calculation : (PlayerColor -> Ball -> Stat list)