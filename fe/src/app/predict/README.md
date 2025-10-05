# Predict Page - Modular Structure

This directory contains the refactored prediction page with a clean, modular architecture.

## 📁 Directory Structure

```
predict/
├── page.tsx                    # Main page component (entry point)
├── page.old.tsx               # Backup of original monolithic file
├── types.ts                   # TypeScript type definitions
├── README.md                  # This file
│
├── components/                # React components
│   ├── ModelSelection.tsx     # Step 1: Model selection UI
│   ├── DataInput.tsx          # Step 2: Data input (single/batch)
│   ├── ResultsTable.tsx       # Step 3: Results table with pagination
│   └── PlanetTypeClassification.tsx  # Step 3: KNN classification chart
│
├── hooks/                     # Custom React hooks
│   ├── usePrediction.ts       # Prediction logic and API calls
│   └── usePlanetTypeClassification.ts  # Planet type classification
│
└── utils/                     # Utility functions
    └── csvParser.ts           # CSV parsing utilities
```

## 🎯 Component Breakdown

### **Main Page** (`page.tsx`)

- **Size**: ~350 lines (down from 1,322 lines!)
- **Responsibilities**:
  - Page layout and navigation
  - Stepper management
  - State orchestration
  - Routing between steps

### **Types** (`types.ts`)

- Centralized TypeScript definitions
- Shared interfaces and constants
- `REQUIRED_FEATURES` constant for feature definitions

### **Components**

#### 1. `ModelSelection.tsx`

- **Purpose**: Step 1 - Choose or upload a model
- **Props**: `preTrainedModels`, `selectedModel`, `onModelSelect`
- **Features**:
  - Dropdown for pre-trained models
  - Upload tab for custom models
  - Clean card-based UI

#### 2. `DataInput.tsx`

- **Purpose**: Step 2 - Input prediction data
- **Props**: `predictionType`, `singleFeatures`, `metadata`, `uploadedFile`, etc.
- **Features**:
  - Tabbed interface (Single/Batch)
  - TOI metadata inputs
  - 12 feature input fields
  - CSV drag-and-drop upload
  - File preview and removal

#### 3. `ResultsTable.tsx`

- **Purpose**: Step 3 - Display prediction results
- **Props**: `predictionResults`
- **Features**:
  - Search by TOI/TOIPFX
  - Pagination (10 items per page)
  - Probability progress bars
  - Prediction badges
  - Summary statistics

#### 4. `PlanetTypeClassification.tsx`

- **Purpose**: Step 3 - Show KNN classification
- **Props**: `planetTypeChart`, `planetTypeClassifications`
- **Features**:
  - Base64 chart display
  - AI agent panel
  - Classification summary
  - Type legend

### **Hooks**

#### 1. `usePrediction.ts`

- **Purpose**: Handle prediction API calls
- **Returns**: `{ isLoading, predictionResults, handlePredict }`
- **Features**:
  - Single/batch prediction logic
  - CSV metadata extraction
  - Error handling
  - Loading states

#### 2. `usePlanetTypeClassification.ts`

- **Purpose**: Handle planet type classification
- **Returns**: `{ planetTypeChart, planetTypeClassifications, fetchPlanetTypeClassifications }`
- **Features**:
  - KNN classification API call
  - Chart and data management
  - Non-blocking error handling

### **Utils**

#### `csvParser.ts`

- **Functions**:
  - `findHeaderRow()`: Finds CSV header (skips comments)
  - `parseCSVMetadata()`: Extracts TOI/TOIPFX
  - `parseCSVFeatures()`: Extracts feature columns
- **Features**:
  - Handles comment lines (`#`)
  - Skips empty lines
  - Robust error handling

## 🔄 Data Flow

```
User Action
    ↓
Main Page (page.tsx)
    ↓
├─→ ModelSelection Component
├─→ DataInput Component
│       ↓
│   usePrediction Hook
│       ↓
│   csvParser Utils
│       ↓
│   API Call (/predict)
│       ↓
│   usePlanetTypeClassification Hook
│       ↓
│   API Call (/classify/planet-types)
│       ↓
└─→ ResultsTable Component
└─→ PlanetTypeClassification Component
```

## ✨ Benefits of Modular Structure

### **1. Maintainability**

- ✅ Each component has a single responsibility
- ✅ Easy to locate and fix bugs
- ✅ Clear separation of concerns

### **2. Reusability**

- ✅ Components can be used in other pages
- ✅ Hooks can be shared across features
- ✅ Utils are generic and testable

### **3. Readability**

- ✅ 350 lines vs 1,322 lines in main file
- ✅ Self-documenting component names
- ✅ Clear file organization

### **4. Testability**

- ✅ Each component can be tested in isolation
- ✅ Hooks can be tested independently
- ✅ Utils are pure functions

### **5. Scalability**

- ✅ Easy to add new steps
- ✅ Simple to extend functionality
- ✅ Clear patterns to follow

## 🚀 Usage

### **Adding a New Step**

1. Create a new component in `components/`
2. Add step to `steps` array in `page.tsx`
3. Add case to `renderStepContent()` switch statement

### **Adding a New Feature**

1. Add types to `types.ts`
2. Create hook in `hooks/` if needed
3. Create component in `components/`
4. Import and use in `page.tsx`

### **Modifying CSV Parsing**

1. Edit functions in `utils/csvParser.ts`
2. Changes automatically apply to all consumers

## 📝 Key Patterns

### **Component Props Pattern**

```typescript
interface ComponentProps {
  // Data
  data: DataType;

  // Handlers
  onAction: (param: Type) => void;

  // State
  isLoading?: boolean;
}
```

### **Hook Pattern**

```typescript
export function useFeature() {
  const [state, setState] = useState();

  const handleAction = async () => {
    // Logic here
  };

  return { state, handleAction };
}
```

### **Util Pattern**

```typescript
export function utilFunction(input: Type): ReturnType {
  // Pure function logic
  return result;
}
```

## 🔧 Environment Variables

Required in `.env.local`:

- `NEXT_PUBLIC_API_ENDPOINT` - Main prediction API
- `NEXT_PUBLIC_API_URL` - Planet type classification API

## 📦 Dependencies

- React 18+
- Next.js 14+
- Tailwind CSS
- Radix UI components
- Lucide React icons

## 🐛 Troubleshooting

### **Component not rendering**

- Check imports in `page.tsx`
- Verify props are passed correctly
- Check console for errors

### **Hook not working**

- Ensure hook is called at component top level
- Check API endpoints are correct
- Verify environment variables

### **CSV parsing fails**

- Check file format (UTF-8)
- Verify header row exists
- Ensure required columns present

## 📚 Further Reading

- [React Hooks Documentation](https://react.dev/reference/react)
- [Next.js App Router](https://nextjs.org/docs/app)
- [TypeScript Best Practices](https://www.typescriptlang.org/docs/)

---

**Last Updated**: 2025-10-05  
**Original File Size**: 1,322 lines  
**New Main File Size**: ~350 lines  
**Reduction**: 73% smaller! 🎉
