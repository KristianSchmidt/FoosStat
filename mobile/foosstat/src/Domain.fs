namespace App

module Domain =
    type Team =
        | SingleTeam of string
        | DoubleTeam of string * string

        override this.ToString() = match this with
                                   | SingleTeam(s) -> s
                                   | DoubleTeam(s1,s2) -> sprintf "%s/%s" s1 s2

        member this.name = this.ToString()

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

        member this.asString = this.ToString()

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

module Analytics =
    open Domain

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

    let fiveBarStealInit color = function | Possession(c, Midfield) when c <> color -> true | _ -> false
    let fiveBarStealSucc color = function | Possession(c, Midfield) when c = color -> true | _ -> false

    let twoBarClearsInit color = function | Possession(c, Defence) when c = color -> true | _ -> false
    let twoBarClearsSucc color = function
                                 | Possession(c, Midfield) when c = color -> true
                                 | Possession(c, Attack) when c = color -> true
                                 | Goal(c) when c = color -> true
                                 | _ -> false

    let matchStat name init succ tryfail =
        let calcFunc color ball =
            let stat = tryfail (init color) (succ color)
            ball |> pairwiseStat stat
        MatchStat(name, calcFunc)

    let goals                = matchStat "Goals"                 (fun _ _ -> true) threeBarSucc genericNumber
    let threeBarGoals        = matchStat "Three bar goals/shots" threeBarInit threeBarSucc genericTryFail
    let threeBarGoalsRecatch = matchStat "Three bar goals/poss"  threeBarInit threeBarSucc genericTryFailRecatch
    let fiveBarPasses        = matchStat "Five bar passes/atts"  midfieldInit midfieldSucc genericTryFail
    let fiveBarPassesRecatch = matchStat "Five bar passes/poss"  midfieldInit midfieldSucc genericTryFailRecatch
    let twoBarGoals          = matchStat "Two bar goals"         twoBarGoalInit twoBarGoalSucc genericNumber
    let fiveBarSteals        = matchStat "Five bar steals"       fiveBarStealInit fiveBarStealSucc genericTryFailRecatch
    let twoBarClears         = matchStat "Two bar clears"        twoBarClearsInit twoBarClearsSucc genericTryFailRecatch

    let gameStats = [goals; twoBarGoals; threeBarGoals; threeBarGoalsRecatch; fiveBarPasses; fiveBarPassesRecatch; fiveBarSteals; twoBarClears]

    let matchSummary game = gameStats |> List.map (generateMatchSummary game)

module Parser = 
    open System
    open Domain

    let parseEvent = function
        | "r2" -> Possession(Red, Defence)
        | "r5" -> Possession(Red, Midfield)
        | "r3" -> Possession(Red, Attack)
        | "b2" -> Possession(Blue, Defence)
        | "b5" -> Possession(Blue, Midfield)
        | "b3" -> Possession(Blue, Attack)
        | "g_r" -> Goal(Red)
        | "g_b" -> Goal(Blue)
        | x -> failwith (sprintf "Non specified event encountered: \"%s\"" x)

    let parseBall (ballLine : string) = 
        let eventStrings = ballLine.Split [|','|]
        let events = eventStrings |> Array.map parseEvent
        Ball(List.ofArray events)

    let parseSet (setLines : seq<string>) = Set(List.ofSeq (Seq.map parseBall setLines))

    let parseGame (text : string) =
        let lines = text.Split([|"\r\n"|], StringSplitOptions.RemoveEmptyEntries)
        let isSetLine (lineNumber : int) (line : string) =
            match line.StartsWith("set") with
            | true -> Some(lineNumber)
            | false -> None

        let parseName prefix =
            lines
            |> Array.tryFind (fun s -> s.StartsWith prefix)
            |> Option.map (fun s -> s.Replace(prefix,"").Split([|'/'|]))
            |> Option.bind (fun arr -> if (arr.Length = 1) then
                                            SingleTeam arr.[0] |> Some
                                       else if (arr.Length = 2) then
                                            DoubleTeam(arr.[0],arr.[1]) |> Some
                                       else
                                            None
                           )

        let indicesBase = Array.mapi isSetLine lines |> Array.choose id
        let indices = Array.append indicesBase [| lines.Length |]
        let intervals = seq { for i = 0 to indices.Length - 2 do yield (indices.[i],indices.[i+1]) }
        let sets =            
            intervals
            |> Seq.map (fun (a,b) -> seq { for i = a+1 to b-1 do yield lines.[i] })
            |> Seq.map parseSet
            |> List.ofSeq

        let blue = defaultArg (parseName "BLUE:") (SingleTeam "Blue")
        let red  = defaultArg (parseName "RED:") (SingleTeam "Red")
        { Sets = sets; Red = red; Blue = blue }
