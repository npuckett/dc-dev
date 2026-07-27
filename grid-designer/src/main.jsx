import React from 'react'
import ReactDOM from 'react-dom/client'
// v3's <AppV3/> (src/v3/AppV3.jsx) is kept, untouched and unmounted, as the
// record of the retired surface-fit approach — see HANDOFF.md §0 and V4_SPEC.md
// §7. This is the v4 pivot's entry point.
import AppV4 from './v4/AppV4.jsx'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppV4 />
  </React.StrictMode>,
)
