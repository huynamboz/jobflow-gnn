import { Navigate, Route, Routes } from "react-router-dom";

import { AdminRoute } from "@/components/admin-route";
import { PublicRoute } from "@/components/public-route";

import LoginPage from "@/pages/login";
import DashboardPage from "@/pages/admin/dashboard";
import LabelingPage from "@/pages/admin/labeling";
import ModelsPage from "@/pages/admin/models";
import JobsPage from "@/pages/admin/jobs";
import CVsPage from "@/pages/admin/cvs";
import RecommendPage from "@/pages/admin/recommend";

function App() {
  return (
    <Routes>
      <Route element={<AdminRoute />} path="/admin">
        <Route index element={<DashboardPage />} />
        <Route path="labeling" element={<LabelingPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="cvs" element={<CVsPage />} />
        <Route path="recommend" element={<RecommendPage />} />
      </Route>

      <Route element={<PublicRoute />}>
        <Route path="/login" element={<LoginPage />} />
      </Route>

      <Route path="*" element={<Navigate replace to="/admin" />} />
    </Routes>
  );
}

export default App;
