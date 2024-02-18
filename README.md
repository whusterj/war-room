# Exploring Probabilities in War Room by Larry Harris

War Room is a board game by Larry Harris, the designer of Axis and Allies. I see it as at least a spiritual successor to AA, if not a complete overhaul. Like AA, it's a grand strategy game set during World War II. Players control the major Axis and Allied powers and vie for the control of territories. The ultimate goals is to capture the opposing capitals and win the war.

See [main.ipynb](main.ipynb) for the exploration.

I previously did an exploration of battle outcome probabilities in Axis and Allies using JavaScript and C (links below). Battles in War Room are similar in some ways and very different in others. The unit types are mostly identical, but the way dice are rolled and hits are allocated is different. War Room's dice have six different colors distributed across 12 faces. The first four colors: red, green, blue, and yellow correspond to different unit types. The last two colors: black and white, are wild, and may or may not be applied in specific circumstances.

The other big difference from AA is that only one "round" of combat and dice rolling happens per combat per turn. The impact of this rule is that most combats end up playing out over several turns, during which time the territory remains "contested," and players can make plans and moves that reinforce their forces in the territory. I personally like this. I think it reflects the reality of having different "fronts" open in a given region and potentially bogging down as both sides throw more and more resources into the fight.

## Links

- [Codepen: War Room Air/Ground Battle Board Simulator](https://codepen.io/whusterj/full/YzgJdEy/9439d70207e540169361fbbb0e6133e1) - JavaScript implementation of War Room air/ground battle board. Select attacking and defending units and see what might happen.
- [(2020) Explore JavaScript with Axis & Allies](https://williamhuster.com/explore-js-with-axis-and-allies/) - Blog post I wrote breaking down the JavaScript code I wrote to simulate Axis and Allies battles.
- [Codepen: Axis and Allies in JavaScript](https://codepen.io/whusterj/pen/VwvjzQv/b4397c0d26fc315dae283d682f7819d8) - Complete JavaScript implementation of Monte Carlo simulation of Axis and Allies battles.
- [Github: Axis and Allies in C](https://github.com/whusterj/axis-and-allies) - Naive implementation of Axis and Allies battle simulator in C. Tinkering with a focus on execution speed.
- [Board Game Geek: War Room](https://boardgamegeek.com/boardgame/229713/war-room)
