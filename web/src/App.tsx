import { Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { isActive, JobList, useJobs } from './components/Jobs'
import { GlossaryPanel, openGlossary } from './components/Glossary'
import { Icon, PageHead } from './components/ui'
import Datasets from './pages/Datasets'
import Diff from './pages/Diff'
import GraphPage from './pages/GraphPage'
import HowItWorks from './pages/HowItWorks'
import Heatmap from './pages/Heatmap'
import QuestionPage from './pages/QuestionPage'
import Results from './pages/Results'
import RunPage from './pages/RunPage'
import Runs from './pages/Runs'
import Targets from './pages/Targets'

// The sidebar follows the workflow: define data and targets, build the two
// kinds of pipeline, then read the benchmark results.
export default function App() {
  const jobs = useJobs()
  const active = jobs.filter(isActive).length
  const item = (to: string, icon: React.ReactNode, label: string, count?: number) => (
    <NavLink to={to} className={({ isActive: a }) => `nav-item ${a ? 'active' : ''}`}>{icon}{label}{count ? <span className="count">{count}</span> : null}</NavLink>
  )
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="logo"><span className="logo-mark">R</span>rag-lab</div>
        {item('/how-it-works', Icon.help, 'How it works')}
        <div className="nav-group">Define</div>
        {item('/datasets', Icon.data, 'Datasets')}
        {item('/targets', Icon.target, 'Targets')}
        <div className="nav-group">Pipelines</div>
        {item('/ingestion', Icon.ingest, 'Ingestion')}
        {item('/retrieval', Icon.retrieve, 'Retrieval')}
        <div className="nav-group">Benchmark</div>
        {item('/results', Icon.results, 'Results')}
        {item('/runs', Icon.runs, 'Runs')}
        {item('/diff', Icon.diff, 'Compare')}
        {item('/heatmap', Icon.heat, 'By category')}
        <div className="nav-group">System</div>
        {item('/jobs', Icon.jobs, 'Jobs', active)}
        <a className="nav-item" href="#" onClick={e => { e.preventDefault(); openGlossary() }} title="Every metric and term explained">{Icon.help}Glossary</a>
        <div className="sidebar-foot">Local playground · scripts run on this machine</div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/datasets" replace />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/:name" element={<Datasets />} />
          <Route path="/targets/:id?" element={<Targets />} />
          <Route path="/ingestion/:id?" element={<GraphPage key="ingestion" kind="ingestion" />} />
          <Route path="/retrieval/:id?" element={<GraphPage key="retrieval" kind="retrieval" />} />
          <Route path="/results" element={<Results />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/runs/:runId/q/:qid" element={<QuestionPage />} />
          <Route path="/heatmap" element={<Heatmap />} />
          <Route path="/diff" element={<Diff />} />
          <Route path="/jobs" element={<><PageHead title="Jobs" sub="Dataset ingests, ingestion pipeline runs and benchmarks run one at a time in the background." /><JobList jobs={jobs} limit={50} /></>} />
        </Routes>
      </main>
      <GlossaryPanel />
    </div>
  )
}
