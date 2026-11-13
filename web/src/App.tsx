import { NavLink, Route, Routes } from 'react-router-dom'
import Runs from './pages/Runs'
import RunPage from './pages/RunPage'
import Heatmap from './pages/Heatmap'
import Diff from './pages/Diff'
import QuestionPage from './pages/QuestionPage'
import Datasets from './pages/Datasets'

export default function App() {
  return (
    <>
      <header className="topbar">
        <span className="brand">rag-lab</span>
        <nav>
          <NavLink to="/" end>Runs</NavLink>
          <NavLink to="/heatmap">Heatmap</NavLink>
          <NavLink to="/diff">Diff</NavLink>
          <NavLink to="/datasets">Datasets</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Runs />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/runs/:runId/q/:qid" element={<QuestionPage />} />
          <Route path="/heatmap" element={<Heatmap />} />
          <Route path="/diff" element={<Diff />} />
          <Route path="/datasets" element={<Datasets />} />
        </Routes>
      </main>
    </>
  )
}
