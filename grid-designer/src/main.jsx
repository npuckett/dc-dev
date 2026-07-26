import React from 'react'
import ReactDOM from 'react-dom/client'
// v2's <App/> (src/App.jsx) is kept, untouched, for reference — see
// V3_SPEC.md §8 P5 and HANDOFF.md. This is the v3 pivot's entry point.
import AppV3 from './v3/AppV3.jsx'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppV3 />
  </React.StrictMode>,
)
