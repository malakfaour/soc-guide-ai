import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AlertProvider } from './lib/store';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Triage from './pages/Triage';
import Remediation from './pages/Remediation';
import Analytics from './pages/Analytics';

export default function App() {
  return (
    <AlertProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="triage" element={<Triage />} />
            <Route path="remediation" element={<Remediation />} />
            <Route path="analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AlertProvider>
  );
}
