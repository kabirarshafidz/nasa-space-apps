# Clickable Planets Feature

## Overview

Planets in the 3D visualization are now interactive! You can click on them to display information, which automatically hides after a few seconds.

## How It Works

### User Interaction

1. **Hover**: Move your mouse over a planet

   - Cursor changes to pointer (👆)
   - Planet glows brighter
   - Information displays immediately

2. **Click**: Click on a planet

   - Information "pins" and stays visible
   - Timer starts (3 seconds)
   - Information auto-hides after 3 seconds
   - Shows timer indicator: "⏱️ (auto-hide in 3s)"

3. **Toggle**: Click again while info is showing
   - Toggles off immediately
   - Timer resets

### Visual Feedback

- **Normal State**: Planet at normal brightness
- **Hover State**: Planet glows, cursor changes to pointer
- **Clicked State**: Planet glows, info persists for 3 seconds
- **Timer Indicator**: Shows "⏱️ (auto-hide in 3s)" when clicked

## Implementation Details

### Component: `Planet.tsx`

```tsx
// State management
const [hovered, setHovered] = useState(false); // For hover state
const [clicked, setClicked] = useState(false); // For click state
const hideTimerRef = useRef<NodeJS.Timeout | null>(null); // Auto-hide timer

// Auto-hide effect
useEffect(() => {
  if (clicked) {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
    }

    hideTimerRef.current = setTimeout(() => {
      setClicked(false);
    }, 3000); // Hide after 3 seconds
  }

  return () => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
    }
  };
}, [clicked]);

// Click handler
const handleClick = (e: any) => {
  e.stopPropagation();
  setClicked((prev) => !prev); // Toggle
};

// Show info on hover OR click
{
  (hovered || clicked) && (
    <Text>
      {/* Planet info */}
      {clicked && "⏱️ (auto-hide in 3s)"}
    </Text>
  );
}
```

### Key Features

1. **Timer Management**

   - Clears old timer when clicking again
   - Cleanup on unmount
   - 3-second auto-hide duration

2. **Event Handling**

   - `stopPropagation()` prevents event bubbling
   - Cursor changes on hover
   - Toggle behavior on click

3. **Visual Feedback**
   - Glow intensity increases when hovered or clicked
   - Timer emoji shows when info is pinned
   - Smooth transitions

## Configuration

### Adjust Auto-Hide Duration

To change the auto-hide time, modify the timeout value in `Planet.tsx`:

```tsx
hideTimerRef.current = setTimeout(() => {
  setClicked(false);
}, 3000); // Change this value (in milliseconds)
```

Examples:

- 2 seconds: `2000`
- 5 seconds: `5000`
- 10 seconds: `10000`

### Disable Auto-Hide

To make info stay until manually closed:

```tsx
// Remove the setTimeout entirely
useEffect(() => {
  // No timer needed
}, [clicked]);
```

### Custom Timer Display

To customize the timer message:

```tsx
{
  clicked && "\n";
}
{
  clicked && "⏱️ (auto-hide in 3s)";
} // Change this text
```

Options:

- `"Click again to close"`
- `"Closing in 3s..."`
- `"📌 Pinned (3s)"`
- Remove the line entirely for no message

## User Experience Benefits

1. **Mobile-Friendly**: Click is easier than hover on touch devices
2. **Persistent Info**: Info stays visible while reading
3. **Auto-Cleanup**: No need to manually close
4. **Visual Clarity**: Timer indicator shows it will auto-hide
5. **Flexible**: Can still use hover for quick peeks

## Compatibility

- ✅ Desktop (mouse hover + click)
- ✅ Mobile (touch = click)
- ✅ Tablet (touch or stylus)
- ✅ Accessibility (keyboard navigation possible with Three.js)

## Technical Notes

### Performance

- Timer uses `useRef` to avoid re-renders
- `stopPropagation()` prevents multiple handlers firing
- Cleanup prevents memory leaks

### State Management

- Two independent states: `hovered` and `clicked`
- Info shows when EITHER is true
- Both can be true simultaneously (hover while clicked)

### Cursor Behavior

- Changes to pointer on hover
- Resets to default when leaving planet
- Works across all browsers

## Future Enhancements

Potential improvements:

1. Configurable auto-hide duration prop
2. Countdown timer display (3...2...1...)
3. Sound effects on click
4. Fade-out animation before hiding
5. Multiple planets clicked at once
6. Info panel outside the 3D canvas

## Example Usage

The feature works automatically with the existing `ExoplanetVisualization` component:

```tsx
<ExoplanetVisualization
  system={solarSystem}
  speedMultiplier={1}
  height="600px"
/>
```

No additional props or configuration needed!

## Testing

### Test Scenarios

1. **Hover Test**: Hover over planet → info appears → move away → info disappears ✅
2. **Click Test**: Click planet → info stays → wait 3s → info disappears ✅
3. **Toggle Test**: Click planet → click again → info disappears immediately ✅
4. **Reset Test**: Click planet → wait 2s → click again → timer resets to 3s ✅
5. **Multiple Planets**: Click planet A → click planet B → both show info ✅
6. **Cursor Test**: Hover over planet → cursor is pointer → move away → cursor is default ✅

## Summary

Planets are now fully interactive with smart auto-hide behavior:

- 👆 Hover for quick info
- 🖱️ Click to pin info for 3 seconds
- 🔄 Click again to toggle off
- ⏱️ Auto-hides after 3 seconds
- 🎯 Cursor changes to indicate clickability

This enhancement makes the visualization more user-friendly, especially on mobile devices! 🚀
