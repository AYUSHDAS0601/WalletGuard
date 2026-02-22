import { AlertTriangle, AlertCircle, Info } from "lucide-react";
import { format } from "date-fns";

interface Activity {
  id: string;
  type: string;
  severity: "high" | "medium" | "low";
  description: string;
  timestamp: string;
}

interface SuspiciousActivityListProps {
  activities: Activity[];
}

export function SuspiciousActivityList({ activities }: SuspiciousActivityListProps) {
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "high":
        return <AlertCircle className="text-red-600" size={20} />;
      case "medium":
        return <AlertTriangle className="text-amber-600" size={20} />;
      default:
        return <Info className="text-blue-600" size={20} />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "border-red-200 bg-red-50";
      case "medium":
        return "border-amber-200 bg-amber-50";
      default:
        return "border-blue-200 bg-blue-50";
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-amber-100 text-amber-800";
      default:
        return "bg-blue-100 text-blue-800";
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold">Suspicious Activities</h2>
      </div>

      <div className="p-6">
        {activities.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <Info size={48} className="mx-auto mb-3 text-gray-400" />
            <p>No suspicious activities detected</p>
          </div>
        ) : (
          <div className="space-y-3">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className={`border rounded-lg p-4 ${getSeverityColor(activity.severity)}`}
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getSeverityIcon(activity.severity)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold text-gray-900">{activity.type}</h3>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${getSeverityBadge(activity.severity)}`}>
                        {activity.severity.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 mb-2">{activity.description}</p>
                    <p className="text-xs text-gray-600">
                      {format(new Date(activity.timestamp), "MMM d, yyyy 'at' h:mm a")}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
