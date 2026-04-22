import type { Icon } from "@tabler/icons-react";

import {
  IconBrain,
  IconBriefcase,
  IconClipboardList,
  IconFiles,
  IconLayoutDashboard,
  IconFileStack,
  IconRobot,
  IconScan,
  IconSettings,
  IconSparkles,
  IconTags,
} from "@tabler/icons-react";

export type AdminNavItem = {
  label: string;
  href: string;
  icon: Icon;
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
        { label: "Dashboard", href: "/admin", icon: IconLayoutDashboard },
        { label: "Labeling", href: "/admin/labeling", icon: IconTags },
        { label: "Models", href: "/admin/models", icon: IconBrain },
        { label: "Jobs", href: "/admin/jobs", icon: IconBriefcase },
        { label: "CVs", href: "/admin/cvs", icon: IconFiles },
        { label: "Recommend", href: "/admin/recommend", icon: IconSparkles },
        { label: "LLM Providers", href: "/admin/llm-providers", icon: IconRobot },
        { label: "LLM Logs", href: "/admin/llm-logs", icon: IconClipboardList },
        { label: "JD Extract", href: "/admin/jd-extract", icon: IconScan },
        { label: "JD Batch", href: "/admin/jd-batch", icon: IconFileStack },
      ],
    },
    {
      title: "ACCOUNT",
      items: [
        { label: "Settings", href: "/admin/settings", icon: IconSettings },
      ],
    },
  ] as AdminNavSection[],
};
