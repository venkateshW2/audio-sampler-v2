# UI/UX Design Rules & Implementation Plan
**Audio Intelligence Sampler v2**

---

## **Current Status: ES6 Module Loading Issue Identified**
**Date**: July 25, 2025  
**Problem**: Main index.html stuck at loading due to ES6 module loading failure  
**Working**: debug_index.html with inline JavaScript functions correctly  
**Decision**: **Option A** - Fix ES6 modules (maintain architecture for Electron migration)

---

## **Architecture Foundation**

### **Why ES6 Modules Were Chosen** ✅
1. **Electron Migration Ready**: Phase 3 desktop app requires modular architecture
2. **Professional Structure**: Clean component separation (Timeline, AudioLoader, RegionManager, WaveformPlayer)
3. **Maintainability**: Individual modules easier to debug, test, and enhance
4. **Scalability**: Supports planned intelligent-sampler-daw component structure
5. **Future-Proofing**: Avoid massive refactor when transitioning to desktop

### **Sacred Architecture Compliance**
- **Reference Document**: `/mnt/2w12-data/audio-sampler-architecture.md`
- **Phased Approach**: Browser UI (Phase 1-2) → Electron Desktop (Phase 3)
- **Component Strategy**: Modular design from browser to desktop seamlessly

---

## **UI/UX Design Principles**

### **1. Professional DAW-Style Interface**
- **Three-Panel Layout**: File Management | Waveform Timeline | Analysis Results
- **Dark Theme**: Professional audio production aesthetic
- **Information Density**: Maximum data display without clutter
- **Visual Hierarchy**: Clear distinction between primary/secondary information

### **2. Real-Time Feedback**
- **Progress Indicators**: Loading states during analysis
- **Status Indicators**: Backend connection, engine status, plugin readiness
- **Interactive Elements**: Hover states, click feedback, selection highlighting
- **Error Communication**: Clear error messages with actionable solutions

### **3. Audio-Centric Design**
- **Waveform Prominence**: Central focus on visual audio representation
- **Timeline Integration**: Seamless segment navigation and playback controls
- **Musical Context**: BPM, key, time signature prominently displayed
- **Classification Visual**: Color-coded content type identification

---

## **Current Implementation Analysis**

### **✅ Working Components** (from debug_index.html)
- **Backend Integration**: Health checks, file analysis, waveform data
- **Peaks.js Visualization**: Dual-view waveforms (overview + zoom)
- **File Processing**: Upload and path-based analysis
- **Analysis Display**: Timeline segments, musical properties, classification results
- **Library Loading**: konva.min.js, waveform-data.min.js, peaks-official.js

### **❌ Broken Components** (in main index.html)
- **ES6 Module Loading**: `import('./app.js')` failing to load properly
- **Component Dependencies**: Timeline.js, AudioLoader.js, etc. not accessible
- **Application Initialization**: Stuck at "Checking..." status

---

## **Fix ES6 Module Loading - Implementation Plan**

### **Phase 1: Debug Module Loading Issues**

#### **Step 1: Verify Module MIME Types**
- **Issue**: Server may not serve .js files with correct MIME type
- **Test**: Check if `/ui/components/Timeline.js` loads with `Content-Type: application/javascript`
- **Fix**: Configure FastAPI static file handling for ES6 modules

#### **Step 2: Check Module Import Paths**
- **Current**: `import { Timeline } from './components/Timeline.js'`
- **Verify**: All relative paths resolve correctly from `/ui/app.js`
- **Test**: Direct browser access to component files

#### **Step 3: Browser ES6 Module Support**
- **Requirement**: ES6 modules need `type="module"` script tags
- **Current**: Using dynamic `import()` which may have different requirements
- **Fix**: Ensure proper module loading configuration

#### **Step 4: Dependency Chain Verification**
- **Components**: Timeline → RegionManager → AudioLoader → WaveformPlayer
- **Test**: Each component loads and exports correctly
- **Debug**: Console errors during import resolution

### **Phase 2: Component Architecture Fixes**

#### **Component Structure Validation**
```
ui/
├── app.js                 # Main application (imports all components)
├── components/
│   ├── Timeline.js        # Peaks.js waveform display
│   ├── RegionManager.js   # Segment visualization
│   ├── AudioLoader.js     # File processing
│   └── WaveformPlayer.js  # Audio playback controls
└── libs/                  # External libraries
```

#### **Import/Export Standardization**
- **Export Format**: `export class ComponentName`
- **Import Format**: `import { ComponentName } from './path.js'`
- **Consistency**: All components follow same pattern

### **Phase 3: UI Integration**

#### **DOM Element Mapping**
- **Container IDs**: Ensure `overview-container`, `zoom-container` exist
- **Element Access**: Verify all `getElementById()` targets exist
- **Event Binding**: Confirm event listeners attach to real DOM elements

#### **Initialization Sequence**
1. **DOM Ready**: Wait for complete DOM loading
2. **Peaks.js Libraries**: Verify external library loading
3. **Component Import**: Load ES6 modules successfully
4. **App Initialization**: Create AudioSamplerApp instance
5. **Backend Connection**: Establish health check
6. **UI Activation**: Enable interactive elements

---

## **UI Component Specifications**

### **Header Section**
- **Logo**: Audio Intelligence Sampler v2 with wave icon
- **Status Indicator**: Backend connection status (connected/disconnected)
- **Database Button**: Access to previous analyses

### **Left Panel: File Management**
- **Upload Zone**: Drag-and-drop file upload area
- **Path Input**: Direct file path analysis option
- **System Status**: Engine and plugin readiness indicators
- **Segment List**: Clickable timeline classification segments

### **Center Panel: Waveform Timeline**
- **Transport Controls**: Play/pause/stop with time display
- **Musical Properties**: BPM and key in transport bar
- **Waveform Display**: Dual-view Peaks.js visualization
- **Progress Indicator**: Analysis progress during processing

### **Right Panel: Analysis Results**
- **File Properties**: Duration, path, success status
- **Timeline Classification**: Segment breakdown with confidence scores
- **Musical Analysis**: Tempo, key, time signature details
- **Database Stats**: System information and analysis history

---

## **Error Handling & User Experience**

### **Loading States**
- **Initial Load**: "Initializing..." with spinner
- **File Processing**: Progress bar with descriptive text
- **Backend Communication**: Connection status indicators
- **Component Failures**: Graceful degradation with error messages

### **Error Recovery**
- **Backend Disconnection**: Retry mechanism with user notification
- **File Processing Errors**: Clear error messages with troubleshooting steps
- **Waveform Failures**: Fallback to analysis-only mode
- **Module Loading Issues**: Development mode error reporting

---

## **Performance Optimization**

### **Loading Performance**
- **Progressive Loading**: Core UI first, enhancement features second
- **Module Lazy Loading**: Load components only when needed
- **Asset Optimization**: Minimize library sizes, cache effectively
- **Backend Communication**: Efficient API calls, avoid redundant requests

### **Visual Performance**
- **Waveform Rendering**: Peaks.js optimization for large files
- **DOM Updates**: Batch updates to avoid layout thrashing
- **Memory Management**: Clean up Peaks instances, avoid memory leaks
- **Responsive Design**: Efficient CSS for various screen sizes

---

## **Testing Strategy**

### **Component Testing**
- **Individual Modules**: Test each component in isolation
- **Integration Testing**: Verify component interactions
- **Browser Compatibility**: ES6 module support across browsers
- **Error Scenarios**: Network failures, invalid files, missing dependencies

### **User Experience Testing**
- **File Upload**: Various audio formats and sizes
- **Analysis Results**: Verify accurate data display
- **Waveform Display**: Visual accuracy and interactivity
- **Workflow Testing**: Complete user journey from upload to analysis

---

## **Migration Readiness (Phase 3: Electron)**

### **Electron Compatibility Checklist**
- ✅ **ES6 Modules**: Component architecture ready
- ✅ **File System Access**: Path-based analysis working
- ✅ **Library Integration**: Peaks.js works in browser environment
- ✅ **API Communication**: Backend integration established

### **Desktop Enhancement Opportunities**
- **Native File Dialogs**: Replace web file input
- **Drag-and-Drop**: OS-level file dropping
- **Menu Integration**: Native menu structure
- **Keyboard Shortcuts**: System-level hotkeys

---

## **Immediate Action Items**

### **Priority 1: Fix ES6 Module Loading**
1. **Debug MIME Types**: Ensure .js files served correctly
2. **Test Component Access**: Direct browser loading of component files
3. **Fix Import Paths**: Resolve any path resolution issues
4. **Verify DOM Ready**: Ensure proper initialization sequence

### **Priority 2: Restore Full Functionality**
1. **Component Integration**: All modules working together
2. **UI Polish**: Match debug_index.html functionality
3. **Error Handling**: Robust error recovery
4. **Performance**: Optimize loading and rendering

### **Priority 3: Enhancement**
1. **Visual Polish**: Professional DAW aesthetic
2. **UX Refinement**: Smooth interactions and feedback
3. **Accessibility**: Keyboard navigation, screen reader support
4. **Documentation**: User guides and developer docs

---

## **Success Metrics**

### **Technical Success**
- **Loading Time**: < 3 seconds to interactive state
- **Module Loading**: All ES6 components load successfully
- **Waveform Display**: Peaks.js renders within 2 seconds of analysis
- **Error Rate**: < 1% of user interactions result in errors

### **User Experience Success**
- **Workflow Completion**: Users can complete analysis without confusion
- **Visual Clarity**: Analysis results clearly understood
- **Performance Satisfaction**: No perceived lag in normal operations
- **Error Recovery**: Users can recover from errors without frustration

---

**Last Updated**: July 25, 2025  
**Next Review**: After ES6 module loading fixes implemented  
**Responsible**: Audio Intelligence Sampler v2 Development Team