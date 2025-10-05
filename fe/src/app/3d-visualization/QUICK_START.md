# Quick Start Guide

Get up and running with the 3D Exoplanet Visualization in 5 minutes.

## Prerequisites

✅ Node.js installed  
✅ Project dependencies installed (`npm install`)  
✅ TESS CSV data at `/data/tess.csv`

## Starting the Application

```bash
# From the fe directory
cd /Applications/Documents/programming/nasa/fe

# Start development server
npm run dev

# Open browser to
# http://localhost:3000/3d-visualization
```

## First Time Usage

### 1. Select a Solar System

Click the dropdown in the top-left card labeled "Select Solar System". You'll see systems listed as:

```
TOI-100 (5 planets)
TOI-700 (4 planets)
...
```

Systems with more planets are generally more interesting!

### 2. Explore the 3D View

**Mouse Controls:**

- **Left Click + Drag**: Rotate the view
- **Right Click + Drag**: Pan the view
- **Scroll Wheel**: Zoom in/out

**What You See:**

- 🌟 **Central Star**: Colored by temperature
  - Red = cool star
  - Yellow = sun-like
  - Blue = hot star
- 🌍 **Planets**: Orbiting spheres colored by temperature/habitability
- 🔵 **Orbit Lines**: Circular paths (green if in habitable zone)
- 🟢 **Green Rings**: Habitable zone boundaries

### 3. Interact with Planets

**Hover** over any planet to see:

- TOI identifier
- Radius (in Earth radii)
- Temperature (in Kelvin)
- Orbital distance (in AU)

### 4. Adjust Animation Speed

Use the **Animation Speed** slider to control how fast planets orbit:

- Low (1-50x): Slow, realistic motion
- Medium (50-200x): Balanced view
- High (200-500x): Fast preview

### 5. View Detailed Info

Scroll down to see:

- **Star Properties**: Mass, radius, temperature
- **Habitable Zone Info**: Inner/outer bounds, planets in HZ
- **Planet Cards**: Detailed stats for each planet
  - Green border = in habitable zone
  - ★ = potentially habitable

## Example Systems to Try

### Multi-Planet Systems

Look for systems with 3+ planets to see complex orbital dynamics.

### Habitable Zone Candidates

Systems with green-colored planets or "★ In Habitable Zone" markers.

### Hot Stars vs Cool Stars

Compare:

- Blue/white stars: Large habitable zones
- Red stars: Small, close-in habitable zones

## Understanding the Colors

### Planet Colors

| Color     | Meaning             | Example                      |
| --------- | ------------------- | ---------------------------- |
| 🟢 Green  | In habitable zone   | Potentially has liquid water |
| 🔵 Blue   | Very cold (< 200 K) | Like Neptune                 |
| 🔷 Cyan   | Cold (200-400 K)    | Like Mars                    |
| 🟠 Orange | Warm (400-700 K)    | Like Earth/Venus             |
| 🔴 Red    | Hot (> 700 K)       | Like Mercury/Venus           |

### Star Colors

| Color     | Temperature | Type              |
| --------- | ----------- | ----------------- |
| 🔴 Red    | < 3500 K    | M-dwarf (cool)    |
| 🟠 Orange | 3500-5000 K | K-type            |
| 🟡 Yellow | 5000-6000 K | G-type (Sun-like) |
| ⚪ White  | 6000-7500 K | F-type            |
| 🔵 Blue   | > 7500 K    | A-type (hot)      |

## Common Questions

### Q: Why are some planets missing?

**A:** Planets without complete orbital data (period, stellar parameters) are filtered out to ensure accurate calculations.

### Q: Why do planets move at different speeds?

**A:** This is **Kepler's Second Law** - planets closer to the star move faster. The animation accurately reflects real orbital mechanics!

### Q: What is the habitable zone?

**A:** The region where liquid water could exist on a planet's surface. Too close = too hot, too far = too cold. Shown as green rings.

### Q: Are the sizes accurate?

**A:** Distances are to scale, but planet/star sizes are adjusted for visibility. Real-scale planets would be too small to see!

### Q: Can I export or screenshot?

**A:** Use your browser's screenshot tool. Future versions may include built-in export.

## Keyboard Shortcuts

While focused on the 3D canvas:

- **Mouse Wheel**: Zoom
- **Right Click + Drag**: Pan
- **Left Click + Drag**: Rotate

## Performance Tips

### If the visualization is slow:

1. **Reduce Animation Speed**: Lower values are less CPU intensive
2. **Close Other Tabs**: Free up browser resources
3. **Choose Simpler Systems**: Systems with fewer planets render faster
4. **Use Hardware Acceleration**: Enable in browser settings

### Optimal Settings:

```
Animation Speed: 100-200x
Systems: 2-6 planets
View Distance: Medium zoom (not too close)
```

## Exploring the Data

### Find Interesting Systems

**Earth-like Candidates:**

- Planets in habitable zone (green)
- Radius close to 1 R⊕
- Temperature 250-320 K

**Hot Jupiters:**

- Large planets (> 5 R⊕)
- Very close orbits (< 0.1 AU)
- High temperatures (> 1000 K)

**Compact Systems:**

- Multiple planets < 0.5 AU
- Similar to TRAPPIST-1

## Troubleshooting

### Problem: "No Data Available"

**Solution:**

- Verify `/data/tess.csv` exists
- Check API endpoint: http://localhost:3000/api/tess-data
- Look for console errors (F12)

### Problem: Black screen / no 3D view

**Solution:**

- Check WebGL support: visit https://get.webgl.org
- Update graphics drivers
- Try a different browser (Chrome recommended)

### Problem: Planets not moving

**Solution:**

- Check animation speed slider (may be set to 1x)
- Refresh the page
- Select a different system

### Problem: Page loads slowly

**Solution:**

- First load parses all data (takes a few seconds)
- Subsequent system changes are instant
- Data is cached by browser

## Next Steps

Once you're comfortable:

1. **Compare Systems**: Switch between different stars
2. **Find Patterns**: Notice trends in temperature, size, orbit
3. **Read Documentation**: Check README.md for detailed physics
4. **Learn the Math**: See PHYSICS_REFERENCE.md for formulas

## Tips for Exploration

### Look For:

- ✨ Systems with planets in the habitable zone
- 🔄 Orbital resonances (period ratios like 2:1, 3:2)
- 📏 Size progressions (small inner, large outer or vice versa)
- 🌡️ Temperature gradients across the system

### Compare:

- Sun-like stars vs red dwarfs vs blue giants
- Systems with many planets vs few planets
- Compact systems vs spread-out systems
- Hot vs cold host stars

## Educational Use

This visualization is great for:

- Understanding Kepler's Laws
- Visualizing habitable zones
- Comparing exoplanet systems
- Learning orbital mechanics
- Exploring real TESS data

## Getting Help

### Documentation Files:

- `README.md` - Comprehensive overview
- `PHYSICS_REFERENCE.md` - All formulas explained
- `ARCHITECTURE.md` - System design
- `IMPLEMENTATION_SUMMARY.md` - Build details

### Questions?

Check the console (F12) for error messages or debug info.

## Fun Challenges

Try to find:

- [ ] A system with 5+ planets
- [ ] A planet in the habitable zone
- [ ] A planet smaller than Earth
- [ ] A planet larger than Jupiter (> 11 R⊕)
- [ ] A very hot star (> 8000 K)
- [ ] A very cool star (< 3000 K)
- [ ] A planet with period < 1 day
- [ ] A planet with period > 100 days

---

**Enjoy exploring the universe! 🚀🌌**

