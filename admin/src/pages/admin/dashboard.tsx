import { useEffect, useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Tags, CheckCircle, SkipForward, Clock } from "lucide-react";

import { labelingService } from "@/services/labeling.service";
import type { LabelingStats } from "@/types/labeling.types";

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <Card className="shadow-sm">
      <CardBody className="flex flex-row items-center gap-4 p-4">
        <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${color}`}>
          <Icon className="size-5 text-white" />
        </div>
        <div>
          <p className="text-2xl font-bold text-default-900">{value}</p>
          <p className="text-sm text-default-500">{label}</p>
        </div>
      </CardBody>
    </Card>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<LabelingStats | null>(null);

  useEffect(() => {
    labelingService.getStats().then(setStats).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-default-900">Dashboard</h1>
        <p className="text-default-500">Overview of labeling progress</p>
      </div>

      {stats && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard icon={Tags}        label="Total Pairs"  value={stats.total_pairs} color="bg-blue-500" />
          <StatCard icon={CheckCircle} label="Labeled"      value={stats.labeled}     color="bg-green-500" />
          <StatCard icon={SkipForward} label="Skipped"      value={stats.skipped}     color="bg-yellow-500" />
          <StatCard icon={Clock}       label="Pending"      value={stats.pending}     color="bg-gray-400" />
        </div>
      )}

      {stats && (
        <div className="grid gap-4 sm:grid-cols-2">
          {/* By Reason */}
          <Card className="shadow-sm">
            <CardBody className="p-4">
              <h2 className="mb-3 font-semibold text-default-800">By Selection Reason</h2>
              <div className="space-y-2">
                {Object.entries(stats.by_reason).map(([reason, { labeled, total }]) => (
                  <div key={reason} className="flex items-center justify-between text-sm">
                    <span className="capitalize text-default-600">{reason.replace(/_/g, " ")}</span>
                    <span className="font-medium text-default-800">{labeled} / {total}</span>
                  </div>
                ))}
              </div>
            </CardBody>
          </Card>

          {/* By Split */}
          <Card className="shadow-sm">
            <CardBody className="p-4">
              <h2 className="mb-3 font-semibold text-default-800">By Split</h2>
              <div className="space-y-2">
                {Object.entries(stats.by_split).map(([split, { labeled, total }]) => (
                  <div key={split} className="flex items-center justify-between text-sm">
                    <span className="capitalize text-default-600">{split}</span>
                    <span className="font-medium text-default-800">{labeled} / {total}</span>
                  </div>
                ))}
              </div>
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  );
}
