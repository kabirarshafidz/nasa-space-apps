# Persistent Hover Feature

## Overview

Planets in the 3D visualization have persistent hover information! When you hover over a planet, the information stays visible for 3 seconds after you move your mouse away, giving you time to read the details.

## How It Works

### User Interaction

1. **Hover**: Move your mouse over a planet

   - Cursor changes to pointer (👆)
   - Planet glows brighter
   - Information displays immediately

2. **Move Away**: Move your mouse off the planet

   - Information stays visible
   - Timer starts (3 seconds)
   - Information auto-hides after 3 seconds

3. **Hover Again**: Move back over the planet while info is showing
   - Timer cancels
   - Information stays visible
   - Timer restarts when you move away again

### Visual Feedback

- **Normal State**: Planet at normal brightness
- **Hover State**: Planet glows, cursor changes to pointer, info appears
- **Persist State**: After hover ends, planet glow and info stay visible for 3 seconds
- **Auto-Hide**: Info fades away after 3 seconds of no interaction

## Implementation Details

### Component: `Planet.tsx`

```tsx
// State management
const [showInfo, setShowInfo] = useState(false); // Controls info visibility
const hideTimerRef = useRef<NodeJS.Timeout | null>(null); // Auto-hide timer

// Handle hover start - show info immediately and clear any pending hide timer
const handleHoverStart = (e: any) => {
  e.stopPropagation();
  document.body.style.cursor = "pointer";

  // Clear any pending hide timer
  if (hideTimerRef.current) {
    clearTimeout(hideTimerRef.current);
    hideTimerRef.current = null;
  }

  setShowInfo(true);
};

// Handle hover end - start 3 second timer to hide info
const handleHoverEnd = (e: any) => {
  e.stopPropagation();
  document.body.style.cursor = "default";

  // Clear any existing timer
  if (hideTimerRef.current) {
    clearTimeout(hideTimerRef.current);
  }

  // Set new timer to hide after 3 seconds
  hideTimerRef.current = setTimeout(() => {
    setShowInfo(false);
  }, 3000);
};

// Cleanup on unmount
useEffect(() => {
  return () => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
    }
  };
}, []);

// Show info when active
{
  showInfo && (
    <Text ref={textRef}>
      {/* Planet info - follows planet via textRef position updates */}
    </Text>
  );
}
```

### Key Features

1. **Timer Management**

   - Clears timer when hovering again
   - Cleanup on unmount
   - 3-second auto-hide duration after hover ends

2. **Event Handling**

   - `stopPropagation()` prevents event bubbling
   - Cursor changes on hover
   - Timer starts on hover end

3. **Position Tracking**
   - Text position updates every frame to follow planet
   - Uses `textRef` to update position without re-rendering
   - Smooth tracking as planet orbits

## Configuration

### Adjust Auto-Hide Duration

To change the auto-hide time, modify the timeout value in `handleHoverEnd` function:

```tsx
hideTimerRef.current = setTimeout(() => {
  setShowInfo(false);
}, 3000); // Change this value (in milliseconds)
```

Examples:

- 2 seconds: `2000`
- 5 seconds: `5000`
- 10 seconds: `10000`

### Disable Auto-Hide

To make info stay visible while hovering only (no persistence):

```tsx
// In handleHoverEnd, remove the setTimeout:
const handleHoverEnd = (e: any) => {
  e.stopPropagation();
  document.body.style.cursor = "default";
  setShowInfo(false); // Hide immediately
};
```

## User Experience Benefits

1. **Read-Friendly**: Info stays visible for 3 seconds after hover ends, giving you time to read
2. **No Accidental Loss**: Moving mouse slightly won't hide info immediately
3. **Auto-Cleanup**: Info disappears automatically, no manual closing needed
4. **Smooth Interaction**: Re-hovering cancels the hide timer
5. **Mobile-Friendly**: Works on touch devices (touch = hover)

## Compatibility

- ✅ Desktop (mouse hover)
- ✅ Mobile (touch = hover)
- ✅ Tablet (touch or stylus)
- ✅ Accessibility (keyboard navigation possible with Three.js)

## Technical Notes

### Performance

- Timer uses `useRef` to avoid re-renders
- `stopPropagation()` prevents multiple handlers firing
- Cleanup prevents memory leaks
- Position updates use ref for efficient tracking

### State Management

- Single state: `showInfo` controls all visibility
- Set to `true` on hover start
- Set to `false` after 3 second timer on hover end
- Timer cancelled if hover starts again before expiry

### Cursor Behavior

- Changes to pointer on hover
- Resets to default when leaving planet
- Works across all browsers

## Future Enhancements

Potential improvements:

1. Configurable auto-hide duration prop
2. Countdown timer display (3...2...1...)
3. Fade-out animation before hiding
4. Multiple info display modes (compact/detailed)
5. Pin/unpin toggle for manual control

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

1. **Hover Test**: Hover over planet → info appears → move away → info stays for 3s → disappears ✅
2. **Re-Hover Test**: Hover → move away → hover again within 3s → timer cancels → info stays ✅
3. **Multiple Planets**: Hover planet A → hover planet B → both show info independently ✅
4. **Timer Test**: Hover → move away → wait 3s → info disappears ✅
5. **Cursor Test**: Hover over planet → cursor is pointer → move away → cursor is default ✅
6. **Following Test**: Hover → info follows planet as it orbits ✅

## Summary

Planets now have persistent hover information with smart behavior:

- 👆 Hover to see info instantly
- ⏱️ Info persists for 3 seconds after you move away
- 🔄 Hover again to cancel the hide timer
- 🎯 Cursor changes to indicate interactivity
- 📍 Info follows the planet as it orbits

This enhancement makes the visualization more user-friendly by giving you time to read the information without having to keep your cursor perfectly still! 🚀
