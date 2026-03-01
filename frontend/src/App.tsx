import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import SampleList from './pages/SampleList'
import SampleDetail from './pages/SampleDetail'
import Predictions from './pages/Predictions'
import Attribution from './pages/Attribution'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="samples" element={<SampleList />} />
        <Route path="samples/:id" element={<SampleDetail />} />
        <Route path="predictions" element={<Predictions />} />
        <Route path="attribution/:id" element={<Attribution />} />
      </Route>
    </Routes>
  )
}

export default App
