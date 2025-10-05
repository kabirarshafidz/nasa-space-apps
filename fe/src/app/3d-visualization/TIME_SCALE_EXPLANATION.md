# Time Scale Implementation

## The Problem

Real exoplanet orbital periods range from **hours to years**. Even the fastest planets:

- Hot Jupiter: ~1-3 days
- Earth: 365 days
- Distant planets: months to years

When using accurate physics (ω in rad/s), the motion is imperceptible in real-time.

## The Solution: Dual Time Acceleration

### 1. Base Time Scale (Constant)

```typescript
const TIME_SCALE = 86400 * 3; // ~3 days per second (259,200x)
```

**What this means:**

- 1 real second = 3 simulated days
- A planet with 3-day orbit completes 1 orbit per second
- A planet with 30-day orbit completes 1 orbit every ~10 seconds
- A planet with 300-day orbit completes 1 orbit every ~100 seconds

**Why 3 days/sec?**

- Makes motion clearly visible but not too fast
- You can follow individual planets as they orbit
- Good balance for detailed observation

### 2. User Speed Multiplier (Adjustable)

```typescript
const [speedMultiplier, setSpeedMultiplier] = useState(1); // 0.1x to 10x
```

**Allows fine-tuning:**

- 0.1x: Slow motion (0.3 days/sec) - for very detailed observation
- 1.0x: Normal (3 days/sec) - default, good balance
- 10x: Fast (30 days/sec) - faster overview
- 20x: Very fast (60 days/sec) - quick preview

## Combined Time Acceleration

```typescript
const acceleratedTime = state.clock.elapsedTime * TIME_SCALE * speedMultiplier;
const angle = acceleratedTime * planet.angularSpeed;
```

**Total acceleration:**

- At 0.1x: `25,920x` real time
- At 1.0x: `259,200x` real time
- At 10x: `2,592,000x` real time
- At 20x: `5,184,000x` real time

## Examples

### Hot Jupiter (1-day orbit)

```
Real period: 1 day = 86,400 seconds
ω = 2π / 86,400 = 7.27e-5 rad/s

Without time scale:
- Completes orbit in 86,400 seconds (~24 hours)
- Imperceptible motion ❌

With TIME_SCALE (3 days/sec):
- Accelerated time: 1 sec × 259,200 = 259,200 sec
- Rotations: 259,200 / 86,400 = 3 orbits per second ✓
- Visible, smooth motion!
```

### Earth-like (365-day orbit)

```
Real period: 365 days = 31,536,000 seconds
ω = 2π / 31,536,000 = 1.99e-7 rad/s

Without time scale:
- Completes orbit in 365 days
- No visible motion ❌

With TIME_SCALE (3 days/sec):
- Accelerated time: 1 sec × 259,200 = 259,200 sec
- Rotations: 259,200 / 31,536,000 = 0.0082 per second
- Completes orbit in ~122 seconds (~2 minutes) ✓
```

### Distant Planet (1000-day orbit)

```
Real period: 1000 days = 86,400,000 seconds

With TIME_SCALE (3 days/sec):
- Accelerated time: 1 sec × 259,200 = 259,200 sec
- Rotations: 259,200 / 86,400,000 = 0.003 per second
- Completes orbit in ~333 seconds (~5.5 minutes) ✓
- Use 10x speed: ~33 seconds for full orbit
```

## UI Controls

### Speed Slider

- **Range:** 0.1x to 20x
- **Default:** 1.0x
- **Step:** 0.1x

### Visual Feedback

```tsx
<p>{speedMultiplier.toFixed(1)}x speed</p>
<p>(Base: ~3 days/sec)</p>
```

Shows user both the multiplier and base rate.

## Physics Accuracy

**Important:** Time is accelerated for visualization ONLY

- All calculations still use real SI units
- Angular velocity (ω) is accurate rad/s
- Orbital radius is accurate meters
- Only the animation clock is accelerated

**Formula remains physically correct:**

```
angle = ω × t    (rad/s × s = radians)
```

We just make `t` run faster!

## Comparison Table

| System           | Real Orbit | Visible @ 1x     | Visible @ 10x   |
| ---------------- | ---------- | ---------------- | --------------- |
| Hot Jupiter (1d) | 24 hours   | **3 orbits/sec** | 30 orbits/sec   |
| Mercury (88d)    | 88 days    | 1 orbit/29 sec   | 1 orbit/2.9 sec |
| Earth (365d)     | 1 year     | 1 orbit/122 sec  | 1 orbit/12 sec  |
| Mars (687d)      | 1.9 years  | 1 orbit/229 sec  | 1 orbit/23 sec  |
| Jupiter (4333d)  | 12 years   | 1 orbit/1444 sec | 1 orbit/144 sec |

## Benefits

### ✅ Visible Motion

- All planets move perceptibly
- Fast planets orbit smoothly
- Slow planets complete orbits in reasonable time

### ✅ Relative Speeds Preserved

- Kepler's laws still visible
- Inner planets orbit faster
- Outer planets orbit slower
- Period ratios maintained

### ✅ User Control

- Can slow down for observation
- Can speed up for overview
- Adjustable to preference

### ✅ Scientifically Accurate

- Real physics underneath
- Time acceleration is explicit
- Can calculate real periods from display

## Code Structure

```typescript
// Constants
const TIME_SCALE = 86400 * 30; // Base acceleration

// State
const [speedMultiplier, setSpeedMultiplier] = useState(1);

// Animation
useFrame((state) => {
  // Combine base and user time scales
  const acceleratedTime =
    state.clock.elapsedTime * // Real time (s)
    TIME_SCALE * // Base acceleration
    speedMultiplier; // User adjustment

  // Use with real angular velocity
  const angle = acceleratedTime * planet.angularSpeed;

  // Calculate position
  const x = Math.cos(angle) * planet.orbitRadiusScene;
  const z = Math.sin(angle) * planet.orbitRadiusScene;
});
```

## Adjusting Time Scale

To change base speed, modify `TIME_SCALE`:

```typescript
// Slower (better for hot Jupiters)
const TIME_SCALE = 86400 * 10; // 10 days/sec

// Current (balanced)
const TIME_SCALE = 86400 * 30; // 30 days/sec

// Faster (better for distant planets)
const TIME_SCALE = 86400 * 100; // 100 days/sec
```

Current value (30 days/sec) works well for typical TESS exoplanets with periods of 1-100 days.

## Performance Note

Time acceleration has **no performance impact**:

- Still 60 FPS rendering
- Just updates `angle` differently
- No extra calculations
- Same number of frames

The animation is smoother because motion is more visible!

## Future Enhancements

Potential improvements:

- [ ] Adaptive time scale based on system
- [ ] "Real-time" mode toggle
- [ ] Time scale presets (slow/medium/fast)
- [ ] Display current date/time in simulation
- [ ] Pause/play controls
- [ ] Step-by-step animation

## Summary

The dual time acceleration system:

1. ✅ Makes all orbits visible (base TIME_SCALE)
2. ✅ Maintains relative speeds (accurate physics)
3. ✅ Allows user adjustment (speedMultiplier)
4. ✅ Remains scientifically accurate (SI units)

**Result:** Beautiful, smooth orbital motion that's both visually engaging and physically correct! 🚀
