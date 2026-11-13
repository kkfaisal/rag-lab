import { Navigate, NavLink, Route, Routes } from 'react-router-dom'
import Datasets from './pages/Datasets'
import Diff from './pages/Diff'
import Experiments from './pages/Experiments'
import GraphPage from './pages/GraphPage'
import Heatmap from './pages/Heatmap'
import QuestionPage from './pages/QuestionPage'
import RunPage from './pages/RunPage'
import Runs from './pages/Runs'

// The nav follows the workflow: data -> index pipeline -> retrieval setup -> experiments -> results.
export default function App() {
  return (
    <>
      <header className="topbar">
        <span className="brand">rag-lab</span>
        <nav>
          <NavLink to="/datasets">1 · Datasets</NavLink>
          <NavLink to="/pipelines">2 · Pipelines</NavLink>
          <NavLink to="/retrieval">3 · Retrieval</NavLink>
          <NavLink to="/experiments">4 · Experiments</NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/heatmap">Heatmap</NavLink>
          <NavLink to="/diff">Diff</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/pipelines" replace />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/pipelines/:id?" element={<GraphPage key="pipeline" kind="pipeline" />} />
          <Route path="/retrieval/:id?" element={<GraphPage key="retriever" kind="retriever" />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/runs/:runId/q/:qid" element={<QuestionPage />} />
          <Route path="/heatmap" element={<Heatmap />} />
          <Route path="/diff" element={<Diff />} />
        </Routes>
      </main>
    </>
  )
}
