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
import LLMProvidersPage from "@/pages/admin/llm-providers";
import CVUploadPage from "@/pages/admin/cv-upload";
import LLMLogsPage from "@/pages/admin/llm-logs";
import JDExtractPage from "@/pages/admin/jd-extract";
import JDBatchLayout from "@/pages/admin/jd-batch";
import JDBatchOverview from "@/pages/admin/jd-batch/overview";
import JDBatchNew from "@/pages/admin/jd-batch/new";
import JDBatchDetail from "@/pages/admin/jd-batch/detail";

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
        <Route path="llm-providers" element={<LLMProvidersPage />} />
        <Route path="cvs/upload" element={<CVUploadPage />} />
        <Route path="llm-logs" element={<LLMLogsPage />} />
        <Route path="jd-extract" element={<JDExtractPage />} />
        <Route path="jd-batch" element={<JDBatchLayout />}>
          <Route index element={<JDBatchOverview />} />
          <Route path="new" element={<JDBatchNew />} />
          <Route path=":id" element={<JDBatchDetail />} />
        </Route>
      </Route>

      <Route element={<PublicRoute />}>
        <Route path="/login" element={<LoginPage />} />
      </Route>

      <Route path="*" element={<Navigate replace to="/admin" />} />
    </Routes>
  );
}

export default App;
