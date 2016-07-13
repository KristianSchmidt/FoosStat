module Games

    [<Literal>]
    let game = @"RED:Collignon/Atha
BLUE:Sørensen/Toubro
set 1
b5,b5,b3,g_b
r5,r3,r2,b2,g_b
r5,r3,g_r
b5,b2,b5,r5,r3,b2,b3,b2,r2,r2,b2,r2,r3,g_r
b5,b2,b3,r2,b3,g_b
r5,r3,b2,b2,g_b
r5,r3,g_r
b5,r5,b5,g_b
set 2
r5,r3,b2,r5,r3,b2,b2,r2,g_r
b5,b3,r2,b2,r2,r3,r3,g_r
b5,b5,b3,g_b
r5,b5,r2,r3,g_r
b5,b3,g_b
r5,r3,b2,r5,b5,r5,r3,b2,r2,b2,b2,r2,b2,b3,g_b
r5,r5,r3,g_r
b5,b3,g_b
r5,r3,r3,r5,r3,r2,r2,r2,b2,b5,b3,r2,r2,r2,r2,b2,g_b
set 3
r5,r3,g_r
b5,b5,r5,r3,g_r
b5,r5,b2,r3,r5,r3,r3,g_r
b5,b5,b3,r2,r3,b5,b5,b5,b3,r2,r5,r3,b2,r2,b5,r5,r3,b2,r3,b2,r3,b2,b5,b3,g_b
r5,r3,r2,b2,b5,b5,b3,g_b
r5,r3,g_r
b5,b5,b3,r5,b5,r5,r3,b5,b2,b3,r2,r3,g_r
set 4
b5,r2,r3,g_r
b5,b5,r2,b2,b2,r3,g_r
b5,r5,r3,g_r
b5,r3,g_r
b5,r5,r5,r3,r3,r3,b2,r5,r3,g_r
set 5
b5,b5,b3,r2,b2,g_b
r5,r5,b5,r3,r3,g_r
b5,r2,r3,g_r
b5,r5,b2,r2,b2,r2,r3,r3,g_r
b5,b3,r2,r5,b5,r2,r2,b5,b3,b5,b5,r5,b2,g_b
r5,b5,b2,r5,r3,b2,r5,b5,r5,r3,g_r
b5,b5,g_b
r5,b5,b5,r5,r3,b5,b5,b5,b3,g_b
r5,r3,r3,g_r
b5,r2,b5,r2,r5,r3,r3,r3,g_r"

    [<Literal>]
    let game2 = @"RED:Schmidt/Eskildsen
BLUE:Hald-Bjerrum/Savery
set 1
b5,r2,r5,b2,b3,r2,b5,b3,b2,b3,r2,r5,r5,r3,b2,b3,b5,b3,g_b
r5,b5,b3,g_b
r5,r3,g_r
b5,b5,r5,r3,r3,r2,b2,b2,b3,r2,r5,r3,g_r
b5,b3,r2,b2,r2,r3,r3,b2,r5,r3,g_r
b5,b3,g_b
r5,b5,b3,g_b
r5,r3,g_r
b5,b5,b3,g_b
set 2
r5,r3,r3,r3,g_r
b5,b5,r5,r3,g_r
b5,r5,g_r
b5,r2,b2,b3,b2,r2,r2,g_r
b5,b3,r2,b3,g_r
set 3
b5,b3,b5,r2,b3,r2,b3,r2,r3,r3,g_r
b5,r5,r3,b2,g_b
r5,r5,b5,b3,g_b
r5,r2,g_r
b5,b3,g_b
r5,r3,g_r
b5,b5,b3,g_b
r5,r5,b2,g_b
set 4
r5,r3,g_r
b5,r5,b5,r5,r3,b2,b5,b3,g_b
r5,b5,r2,r2,b2,r2,b2,g_b
r5,r3,g_r
b5,b3,b3,b2,r5,r3,b2,b2,b5,b3,b5,b3,b2,b5,b3,g_b
r5,r3,g_r
b5,r2,r3,g_r
b5,b3,g_b
r5,r3,b2,b5,b3,g_b"

    [<Literal>]
    let game3 = @"RED:Bremer/Faurskov
BLUE:Trier/Wonsyld
set 1
r5,b5,b3,g_b
r5,r5,b2,r5,b2,r2,b2,b5,b3,r5,b2,b5,r2,b5,r2,r3,g_r
b5,b5,r5,r3,g_r
b5,b3,r2,r5,r5,r3,g_r
b5,b2,b3,g_b
r5,r3,b2,b5,b3,g_b
r5,r3,g_r
b5,b3,b3,r2,b2,b5,b3,g_b
r5,b5,b5,b3,r5,r3,r3,b2,r2,g_r
set 2
b5,b5,b3,r2,r3,b2,r3,g_r
b5,r5,r3,b2,r2,b3,b3,g_b
r5,r5,b5,b5,b3,b3,g_b
r5,r5,b2,r5,r3,b2,g_b
r5,b5,b5,r5,b2,r5,r3,r2,g_r
b5,b3,b3,g_b
r5,r3,b5,b3,b5,r5,b2,b5,r5,b5,b3,g_b
set 3
r5,b2,r2,b2,b5,b5,b3,r2,b5,b3,b5,b3,r2,b2,r5,r3,r2,r2,r3,g_r
b5,b5,b5,b3,r5,r3,b2,r5,r3,g_r
b5,b3,b2,r3,g_r
b5,r3,r3,r3,b2,b3,g_b
r5,b3,r2,b3,r2,r3,g_r
b5,b3,r2,r2,r2,r3,b2,r2,b5,b3,g_b
r5,b5,r5,r3,b2,b2,r5,r3,r3,r2,r2,b2,r2,r5,r3,b2,b5,b3,g_b
r5,r3,g_r
set 4
b5,b3,b3,r2,b3,r2,r5,r3,b2,r2,b2,r2,b5,r5,r3,b2,r5,b5,b3,r5,b3,b3,r2,b2,b2,g_r
r5,r3,b3,g_b
r5,r3,r3,g_r
b5,r5,r2,r5,b5,b3,r2,b2,b3,b5,b5,b5,b3,b3,g_b
r5,b2,r5,b2,b3,g_b
r5,r3,g_r
b5,b5,b3,g_b
set 5
r5,r5,r3,g_r
b5,r2,b2,r2,g_r
b5,r5,b2,r2,b3,r2,r2,b2,b2,r3,r3,r2,b2,b3,r2,b5,b5,b5,r2,r3,b5,b5,r2,b2,b3,r2,b2,g_b
r5,r3,g_r
b5,b3,b2,r2,r5,r3,b2,b2,r2,b2,b3,g_b
r5,r3,r3,b2,r2,r2,b3,g_b
r5,r3,b2,r2,b2,b3,r2,r2,b2,b2,r5,r2,r3,g_r
b5,r5,r3,b2,g_b
r5,r3,g_r
b5,b3,g_b
r5,r2,g_r
b5,b3,r2,r2,b2,r2,b2,r3,g_r
"