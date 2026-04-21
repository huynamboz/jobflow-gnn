import type { LucideIcon } from "lucide-react";

import {
  Brain,
  Briefcase,
  FileText,
  LayoutDashboard,
  Settings,
  Tags,
} from "lucide-react";

export type AdminNavItem = {
  label: string;
  href: string;
  icon: LucideIcon;
};

export type AdminNavSection = {
  title: string;
  items: AdminNavItem[];
};

export const adminConfig = {
  name: "JobFlow",
  navSections: [
    {
      title: "GENERAL",
      items: [
        { label: "Dashboard", href: "/admin", icon: LayoutDashboard },
        { label: "Labeling", href: "/admin/labeling", icon: Tags },
        { label: "Models", href: "/admin/models", icon: Brain },
        { label: "Jobs", href: "/admin/jobs", icon: Briefcase },
        { label: "CVs", href: "/admin/cvs", icon: FileText },
      ],
    },
    {
      title: "ACCOUNT",
      items: [
        { label: "Settings", href: "/admin/settings", icon: Settings },
      ],
    },
  ] as AdminNavSection[],
};
