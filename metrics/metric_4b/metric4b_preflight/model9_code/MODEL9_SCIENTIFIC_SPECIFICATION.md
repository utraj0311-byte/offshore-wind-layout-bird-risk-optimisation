# Model 9 — Flight Height + Rotor Transit

## 1. Why Model 9 is being created

Model 8 already introduced **individual bird trajectories with meso-scale avoidance**.

In Model 8:

- birds enter the wind farm;
- birds can respond to turbines at meso scale;
- a response can change the later trajectory;
- the changed trajectory can affect downstream turbine encounters;
- the model records **encounters**.

However, an encounter in Model 8 does **not** mean that a bird actually passes through the rotor.

A bird may pass close to a turbine horizontally but:

- fly below the rotor;
- fly above the rotor;
- pass beside the circular rotor disk.

Therefore Model 8 currently measures a broader **turbine encounter**, not actual rotor exposure.

Model 9 is created to add the missing step:

```math
\boxed{ \text{turbine encounter} \rightarrow \text{actual rotor-swept-area transit} }
```

The literature synthesis supports treating flight-height exposure and rotor transit as a stage that comes **before** blade-collision probability. It also supports keeping meso avoidance separate from micro avoidance and blade-strike probability.

---

# 2. Main research question

The main Model 9 question should be:

> **How does adding bird flight height and turbine rotor geometry convert Model 8's layout-dependent meso-scale trajectories into simulated rotor-swept-area transits?**

A second important validation question is:

> **Does rotor-transit exposure provide new layout information compared with Model 8 encounter intensity, or does it simply scale the Model 8 result by approximately the same amount for every layout?**

This second question is important because Model 9 should demonstrate whether the additional vertical/rotor geometry actually changes the optimisation problem.

---

# 3. What Model 9 should NOT try to answer

Model 9 should **not** answer:

> What is the probability that a bird collides with a blade?

That comes later.

Therefore:

```math
\boxed{ \text{rotor transit}\neq\text{collision} }
```

Model 9 stops once it knows whether a simulated bird trajectory passes through the rotor-swept disk.

It should not yet include:

- micro avoidance;
- blade-strike probability;
- blade rotation;
- rotor RPM;
- bird wingspan;
- bird body length;
- blade chord;
- pitch;
- twist;
- detailed blade geometry;
- mortality.

Those belong to the later collision model.

---

# 4. Relation to the previous models

The modelling progression is:

```text
Earlier models
      ↓
simple/artificial bird-risk objectives
      ↓
Shielding model
      ↓
directional geometric exposure
without behavioural response
      ↓
Model 8
      ↓
individual bird trajectories
+ meso avoidance
+ downstream path changes
      ↓
Model 9
      ↓
flight height
+ rotor geometry
+ rotor transit
      ↓
Later model
      ↓
micro avoidance
+ conditional blade-collision probability
      ↓
Final integrated collision-risk model

```

Model 9 therefore makes the transition from:

```math
\text{2-D turbine encounter}
```

to:

```math
\text{3-D rotor exposure}.
```

---

# 5. What should be inherited unchanged from Model 8

Do not redesign Model 8.

Model 9 should inherit:

- individual bird trajectories;
- the same entry conditions;
- the same meso-scale response logic;
- the same response probability;
- the same response zone;
- the same avoidance turn;
- the same relaxation / return-to-flight logic;
- the same downstream interaction logic.

The purpose of Model 9 is **not to test another avoidance model**.

The new question is only:

> Given the trajectory produced by Model 8, does that bird actually pass through a rotor?

---

# 6. Which avoidance scale is used?

Model 9 keeps:

```math
\boxed{\text{meso avoidance}}
```

from Model 8.

It does **not** add:

### Macro avoidance

Whole-farm avoidance before entering the wind farm.

### Micro avoidance

Last-second response to the rotor blades.

The reviewed literature supports keeping meso-scale behaviour before rotor-scale processes and treating micro avoidance as a later stage.

---

# 7. New component 1 — bird flight height

Each simulated bird receives a flight height:

```math
\boxed{z_j}
```

where:

- `j` = bird trajectory;
- `z_j` = flight height of bird `j`.

Instead of assuming all birds fly at the same height, flight height should come from a literature-based flight-height distribution:

```math
\boxed{ z_j\sim f_H(z) }
```

where:

- `f_H(z)` = flight-height distribution;
- `z_j` = height sampled for individual trajectory `j`.

The literature synthesis strongly supports using a **distribution** rather than assuming uniform or fixed rotor-height flight.

---

# 8. Important simplifying assumption

For the first Model 9 version:

> **Each bird receives one flight height when it enters the model, and that height remains constant throughout its wind-farm crossing.**

So:

```math
z_j=\text{constant}
```

for trajectory `j`.

This is a **Model 9 simplifying assumption**, not a conclusion from the literature.

It prevents us from introducing:

- vertical avoidance;
- climbing;
- diving;
- turbulence-dependent height;
- weather-dependent altitude.

Those can be added later if there is a strong reason.

---

# 9. Species choice

Use **one reference flight-height distribution initially**.

Do not start with multiple bird species.

If a species-specific FHD is used, this does **not automatically make the entire Model 9 species-specific**, because the Model 8 meso-response probability and behavioural rules may still be artificial/scenario parameters.

The methodology should therefore say something like:

> A species-specific FHD is used to represent vertical exposure, while the inherited Model 8 meso-response parameters remain scenario assumptions rather than fully species-calibrated behavioural parameters.

---

# 10. New component 2 — rotor geometry

For each turbine we need only basic rotor geometry.

Let:

```math
D
```

be rotor diameter.

Then:

```math
\boxed{ R=\frac D2 }
```

where:

- `R` = rotor radius.

Let:

```math
H
```

be hub height.

The upper rotor tip is:

```math
H+R
```

and the lower tip is:

```math
H-R.
```

So the vertical rotor-swept range is:

```math
\boxed{ H-R\le z\le H+R }
```

This is useful as a first check.

However, this alone is not sufficient because a rotor is **circular**, not rectangular.

---

# 11. Circular rotor-swept area

The rotor should be represented as a **circular disk**.

Side view:

```text
               top tip
                  ↑
               H + R
                  |
              \   |   /
               \  |  /
             ---- O ----   ← hub at H
               /  |  \
              /   |   \
                  |
               H - R
                  ↓
             bottom tip

```

Therefore being at the correct height is necessary but not sufficient.

For example, a bird could be at hub height but pass far sideways from the turbine.

That should **not** count as rotor transit.

---

# 12. Rotor-plane assumption

Model 9 also needs a clearly stated assumption about the **orientation of the rotor plane**.

For the stripped-down Model 9, use one simple fixed convention.

For example:

> **The rotor plane is assumed perpendicular to the reference bird-flight direction.**

The exact convention can be matched to the existing Model 8 coordinate system.

Do not add:

- turbine yaw control;
- wind-dependent rotor orientation;
- changing nacelle direction.

Those would unnecessarily complicate Model 9.

The important thing is that the convention is clearly stated.

---

# 13. Exact definition of rotor transit

Use this definition:

> **A simulated rotor transit occurs when a bird trajectory crosses the plane of a turbine rotor and the crossing point lies inside the circular rotor-swept disk.**

This should be evaluated separately for each:

```math
\text{bird }j
```

and:

```math
\text{turbine }i.
```

A bird–turbine pair should count at most once for one crossing event.

---

# 14. Very simple example

Suppose the rotor spans approximately:

```math
30\text{ m} \le z\le270\text{ m}.
```

### Bird A

Trajectory passes through turbine centre horizontally.

```math
z=100\text{ m}.
```

Bird is inside the vertical rotor range and crosses inside the circular rotor.

Result:

```math
\boxed{\text{rotor transit}}
```

---

### Bird B

Same horizontal path:

```math
z=320\text{ m}.
```

Bird is above the rotor.

Result:

```math
\boxed{\text{no rotor transit}}
```

---

### Bird C

```math
z=150\text{ m}
```

but the bird passes far to the side of the rotor centre.

Result:

```math
\boxed{\text{no rotor transit}}
```

So Model 9 distinguishes cases that Model 8 currently treats more similarly.

---

# 15. Full Model 9 process

The complete logic should be:

```text
Generate / enter bird trajectory
        ↓
Assign flight height z
        ↓
Run existing Model 8 meso behaviour
        ↓
Bird may respond to upstream turbine
        ↓
Trajectory changes
        ↓
Bird approaches another turbine
        ↓
Does trajectory cross rotor plane?
        ↓
          NO
          ↓
    no rotor transit

          YES
          ↓
Is crossing point inside circular rotor disk?
        ↓
      NO → no rotor transit

      YES
        ↓
record one rotor transit
        ↓
continue trajectory through farm
        ↓
repeat for downstream turbines

```

The important feature is:

> An upstream meso response can change whether a later rotor transit occurs.

That preserves the layout-responsive behaviour that made Model 8 useful.

---

# 16. Primary Model 9 metric

The primary metric should be:

```math
\boxed{ E_{RT} = \frac{ \text{total rotor-transit events} }{ \text{total simulated trajectories} } }
```

Call it:

> **rotor-transit intensity**

More formally:

```math
\boxed{ E_{RT} = \frac{ \sum_{j=1}^{N_{\mathrm{traj}}}N_{RT,j} }{ N_{\mathrm{traj}} } }
```

where:

- `N_{\mathrm{traj}}` = total number of simulated bird trajectories;
- `N_{RT,j}` = number of turbine rotors crossed by bird `j`.

## 16.1 Implemented variance-reduced estimator

The physical quantity above is unchanged, but the implementation reports it
in two ways:

- `E_RT` uses the one sampled constant height assigned to each bird and is
  retained as an audit estimator;
- `E_RT_CE` integrates the same frozen discrete FHD over the eligible circular
  height interval conditional on the realised horizontal path.

For lateral rotor-plane offset `l`, the eligible height interval is

```math
I(l)=\left[H-\sqrt{R^2-l^2},\ H+\sqrt{R^2-l^2}\right]
```

when `|l| <= R`. Thus the conditional bird-turbine contribution is

```math
w_{ji}=P\{Z\in I(l_{ji})\},
```

and

```math
\boxed{E_{RT,CE}=\frac{1}{N_{\mathrm{traj}}}\sum_j\sum_i w_{ji}}.
```

Model 8 horizontal trajectories are independent of flight height, so `w_ji`
is the conditional expectation of the sampled transit indicator. The law of
total expectation gives the same target expectation as `E_RT`, and the law of
total variance explains why integrating height reduces Monte Carlo noise. It
does not add avoidance, collision probability, or another biological model.

---

# 17. Example of rotor-transit intensity

Suppose:

```math
N_{\mathrm{traj}}=10,000.
```

Across those birds there are:

```math
850
```

rotor-transit events.

Then:

```math
E_{RT} = \frac{850}{10000} = 0.085.
```

Interpretation:

> There were, on average, 0.085 simulated rotor transits per entering bird trajectory.

This does **not** mean:

> 8.5% collision probability.

It is purely rotor exposure.

---

# 18. Why intensity is the primary metric

One bird can transit more than one rotor.

Example:

```text
Bird 1 → Rotor 2 → Rotor 5 → Rotor 8

```

That gives:

```math
N_{RT,1}=3.
```

Each transit is another opportunity that a later collision model may need to evaluate.

Therefore intensity retains more information than simply asking whether the bird encountered at least one rotor.

The literature synthesis also supports rotor passage as a separate stage before single-transit collision probability.

---

# 19. Optional secondary metric — rotor-transit prevalence

We may also calculate:

```math
\boxed{ P_{RT} = \frac{ \text{trajectories with at least one rotor transit} }{ N_{\mathrm{traj}} } }
```

Example:

10,000 birds.

600 birds transit at least one rotor.

Then:

```math
P_{RT}=0.06.
```

So:

```math
6\%
```

of simulated entering trajectories passed through at least one rotor.

This is useful descriptively.

But the underlying exposure quantity:

```math
\boxed{E_{RT}}
```

should remain the primary optimisation objective. In computation, `E_RT_CE`
is the preferred estimator of that quantity and sampled `E_RT` is retained for
audit.

---

# 20. Optimisation objective

The two optimisation objectives become:

### Energy

```math
\boxed{ \text{Maximise AEP} }
```

where:

AEP = Annual Energy Production.

### Bird exposure

```math
\boxed{ \text{Minimise }E_{RT,CE} }
```

Therefore:

```math
\boxed{ f_1(X)=-AEP(X) }
```

```math
\boxed{ f_2(X)=E_{RT,CE}(X) }
```

where:

```math
X=\{(x_i,y_i)\}_{i=1}^{N}
```

is the turbine layout.

The software supplies NSGA-II with the positive, normalised equivalents
`(AEP_ceiling - AEP) / AEP_loss_reference` and
`E_RT_CE / E_RT_CE_reference`. Positive scaling does not alter Pareto
dominance; raw AEP and exposure values are retained in every output.

---

# 21. What Model 9 is trying to learn

The main scientific/modelling question is not:

> Does flight height reduce the absolute number of encounters?

That is almost guaranteed.

The more important question is:

> **Does adding flight height and rotor geometry alter the relative performance of different turbine layouts?**

For example:

Model 8 may show:

| LayoutEncounter intensity |      |
| ------------------------- | ---- |
| A                         | 0.20 |
| B                         | 0.25 |
| C                         | 0.30 |

Then Model 9 could give:

| LayoutRotor-transit intensity |      |
| ----------------------------- | ---- |
| A                             | 0.06 |
| B                             | 0.04 |
| C                             | 0.07 |

Now the ranking changes.

That would mean Model 9 contributes new layout information.

---

# 22. Important counter-case to test

It is also possible that:

```math
E_{RT}\approx cE_{\mathrm{enc}}
```

for every layout, where `c` is almost constant.

Example:

| Layout`E_{\mathrm{enc}}E_{RT}` |      |       |
| ------------------------------ | ---- | ----- |
| A                              | 0.20 | 0.040 |
| B                              | 0.25 | 0.050 |
| C                              | 0.30 | 0.060 |

Then:

```math
E_{RT}\approx0.2E_{\mathrm{enc}}.
```

In this case, Model 9 has made exposure more physically meaningful but has not changed layout ranking much.

That is **still a valid result**.

Do not force the model to produce a different ranking.

---

# 23. Key Model 9 validation question

Before doing a large optimisation, compare:

```math
\boxed{ E_{\mathrm{enc}} \quad\text{vs}\quad E_{RT} }
```

for several geometrically different layouts.

Ask:

> **Is rotor-transit intensity genuinely more layout-sensitive/different from encounter intensity, or mainly a proportional reduction?**

This is probably the most important validation question in Model 9.

---

# 24. Basic geometry validation tests

Before optimisation, run simple deterministic tests.

## Test 1 — bird below rotor

Horizontal path crosses turbine centre.

```math
z<H-R.
```

Expected:

```math
\boxed{\text{no rotor transit}}
```

---

## Test 2 — bird above rotor

```math
z>H+R.
```

Expected:

```math
\boxed{\text{no rotor transit}}
```

---

## Test 3 — bird through rotor centre

```math
z=H
```

and the horizontal trajectory crosses the rotor centre.

Expected:

```math
\boxed{\text{rotor transit}}
```

---

## Test 4 — correct height but lateral miss

Bird height lies inside rotor range but crosses the rotor plane outside radius `R`.

Expected:

```math
\boxed{\text{no rotor transit}}
```

These should pass before Model 9 is connected to the optimiser.

All four checks, plus circular tangency, coarse-step, multiple-transit,
permutation, Model 8 parity, and FLORIS-coupling tests, pass in the v4 package.

---

# 25. Stochastic validation

Because individual heights are sampled from:

```math
f_H(z),
```

Model 9 contains stochastic variation.

Do not immediately perform a massive Model 8-style batch campaign.

Instead:

1. run several fixed seeds;
2. compare sampled `E_{RT}` with conditional `E_{RT,CE}` and check whether
   their difference is consistent with zero;
3. determine whether remaining trajectory-batch variation is small relative
   to layout differences;
4. increase repeats only if necessary.

The literature synthesis supports reporting uncertainty, but the implementation should remain proportionate to the purpose of this developmental model.

---

# 26. What should stay OUT of Model 9

Do not add:

### Avoidance

- macro avoidance;
- new micro avoidance;
- vertical avoidance.

### Flight behaviour

- changing altitude;
- weather-dependent height;
- wind-dependent height;
- turbulence response.

### Species complexity

- multiple bird species.

### Collision mechanics

- collision probability;
- blade rotation;
- bird length;
- wingspan;
- flight speed for blade collision;
- blade chord;
- pitch;
- twist;
- airfoil thickness.

### Outcome

- mortality;
- fatality rate.

All of these should be saved for later models.

---

# 27. Literature basis

The five-paper synthesis supports several parts of Model 9.

### Flight height

Flight-height distributions are heterogeneous and influence the proportion of birds exposed to the rotor-swept region.

### Rotor geometry

Rotor diameter, radius and vertical placement determine the space through which a bird must pass to be exposed.

### Rotor transit versus collision

Existing CRM structures separate:

```math
\text{number of birds passing through rotor}
```

from:

```math
P(\text{collision}\mid\text{rotor transit}).
```

Therefore rotor transit is a legitimate stopping point.

### Avoidance

Meso-scale response occurs before rotor-scale interaction, while micro avoidance belongs closer to the actual blade interaction.

These findings support the Model 9 sequence without requiring a full collision model.

---

# 28. What is directly literature-supported and what is our assumption?

| ComponentStatus                                        |                                |
| ------------------------------------------------------ | ------------------------------ |
| Flight-height distributions matter                     | Literature-supported           |
| Rotor geometry affects exposure                        | Literature-supported           |
| Circular rotor-swept area matters                      | Literature-supported           |
| Rotor transit is distinct from collision               | Literature-supported           |
| Meso response precedes micro/blade-scale processes     | Literature-supported           |
| Assign one `z` to each trajectory                      | Model 9 modelling assumption   |
| Keep `z` constant through farm                         | Model 9 simplifying assumption |
| Exact Model 8 path → rotor-disk intersection algorithm | Thesis implementation          |
| Rotor-transit intensity `E_{RT}`                       | Thesis-defined metric          |
| Conditional estimator `E_{RT,CE}`                      | Thesis implementation          |
| Fixed rotor-plane orientation                          | Model 9 simplifying assumption |

---

# 29. Contribution wording

Do **not** claim:

> Model 9 is the first model to combine individual bird trajectories, flight height and rotor exposure.

The five reviewed papers do not establish that.

Instead use:

> **Model 9 is a methodological extension of the existing trajectory framework that couples layout-responsive meso-scale trajectories with flight-height exposure and explicit rotor-swept-area transit.**

An even simpler formulation:

> **Model 9 converts Model 8's horizontal turbine encounters into physically more meaningful rotor-transit events.**

This is a safe and defensible contribution.

---

# 30. Why Model 9 matters in the thesis

Model 9 provides the missing link between:

```math
\text{bird movement around turbines}
```

and:

```math
\text{collision modelling at the rotor}.
```

Model 8 answers:

> Where does the bird go after responding to turbines?

Model 9 answers:

> Does that resulting path actually pass through a rotor?

The later model can then answer:

> Given a rotor transit, what is the probability of avoiding or colliding with the blades?

So the progression is:

```math
\boxed{ \text{meso response} \rightarrow \text{rotor transit} \rightarrow \text{collision probability} }
```

---

# 31. Final Model 9 specification

## Proposed name

**Model 9 — Rotor-Transit Exposure Model**

Alternative:

**Model 9 — Flight-Height and Rotor-Transit Model**

---

## Research question

> **How do flight-height distributions and rotor geometry convert layout-dependent meso-scale bird trajectories into simulated rotor-swept-area transits?**

---

## Starting point

Existing Model 8 individual trajectories and meso avoidance.

---

## New inputs

- flight-height distribution `f_H(z)`;
- individual sampled height `z_j`;
- rotor diameter `D`;
- rotor radius `R`;
- hub height `H`;
- fixed rotor-plane orientation convention.

---

## New mechanism

```math
\boxed{ \text{Model 8 trajectory} \rightarrow \text{assign }z_j \rightarrow \text{cross rotor plane?} \rightarrow \text{inside circular rotor?} \rightarrow \text{record rotor transit} }
```

---

## Primary metric

```math
\boxed{ E_{RT} = \frac{\text{total rotor transits}} {\text{number of trajectories}} }
```

**Rotor-transit intensity.**

It is reported as sampled `E_RT` for audit and conditional `E_RT_CE` for the
default optimisation objective. These estimators target the same expected
quantity.

---

## Optional secondary metric

```math
\boxed{ P_{RT} = \frac{\text{trajectories with at least one transit}} {\text{total trajectories}} }
```

**Rotor-transit prevalence.**

---

## Optimisation

```math
\boxed{\text{Maximise AEP}}
```

and:

```math
\boxed{\text{Minimise }E_{RT,CE}}
```

---

## Primary comparison

```math
\boxed{ E_{\mathrm{enc}} \quad\text{versus}\quad E_{RT} }
```

to determine whether the rotor-transit layer actually changes layout discrimination.

---

## Model endpoint

```math
\boxed{\text{rotor transit}}
```

NOT:

```math
\text{blade collision}.
```

---

## Final one-sentence explanation

> **Model 9 retains the individual meso-responsive bird trajectories developed in Model 8, assigns each trajectory a flight height from a literature-based distribution, and tests whether the resulting three-dimensional path crosses the circular rotor-swept area of each turbine; the same rotor-transit intensity is also evaluated by exact conditional integration of flight height for a lower-variance optimisation objective, while actual blade-collision probability is deliberately left for the subsequent model.**
