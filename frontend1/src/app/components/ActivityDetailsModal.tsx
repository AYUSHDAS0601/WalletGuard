import { X, AlertCircle, AlertTriangle, CheckCircle, Clock } from "lucide-react";
import { format } from "date-fns";

interface Activity {
  id: string;
  type: string;
  severity: "high" | "medium" | "low";
  description: string;
  timestamp: string;
  details: string;
}

interface ActivityDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  activities: Activity[];
  address: string;
}

export function ActivityDetailsModal({ isOpen, onClose, activities, address }: ActivityDetailsModalProps) {
  if (!isOpen) return null;

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "high":
        return <AlertCircle className="text-red-600" size={24} />;
      case "medium":
        return <AlertTriangle className="text-amber-600" size={24} />;
      default:
        return <CheckCircle className="text-green-600" size={24} />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "border-red-200 bg-red-50";
      case "medium":
        return "border-amber-200 bg-amber-50";
      default:
        return "border-green-200 bg-green-50";
    }
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case "high":
        return "bg-red-100 text-red-800 border-red-200";
      case "medium":
        return "bg-amber-100 text-amber-800 border-amber-200";
      default:
        return "bg-green-100 text-green-800 border-green-200";
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="relative bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-gray-200 px-8 py-6 flex items-center justify-between z-10">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Activity Details</h2>
              <p className="text-sm text-gray-600 font-mono mt-1">{address}</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X size={24} className="text-gray-600" />
            </button>
          </div>

          {/* Content */}
          <div className="px-8 py-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            <div className="space-y-6">
              {activities.map((activity, index) => (
                <div
                  key={activity.id}
                  className={`border-2 rounded-xl p-6 ${getSeverityColor(activity.severity)}`}
                >
                  {/* Activity Header */}
                  <div className="flex items-start gap-4 mb-4">
                    <div className="flex-shrink-0">
                      {getSeverityIcon(activity.severity)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="text-xl font-bold text-gray-900 mb-1">
                            {activity.type}
                          </h3>
                          <span className={`inline-block text-xs px-3 py-1 rounded-full font-semibold border-2 ${getSeverityBadge(activity.severity)}`}>
                            {activity.severity.toUpperCase()} SEVERITY
                          </span>
                        </div>
                      </div>
                      
                      {/* Short Description */}
                      <div className="bg-white bg-opacity-60 rounded-lg p-3 mb-3 border border-gray-200">
                        <p className="text-sm font-medium text-gray-900">{activity.description}</p>
                      </div>

                      {/* Timestamp */}
                      <div className="flex items-center gap-2 text-sm text-gray-600 mb-4">
                        <Clock size={16} />
                        <span>Detected: {format(new Date(activity.timestamp), "MMMM d, yyyy 'at' h:mm a")}</span>
                      </div>

                      {/* Detailed Information */}
                      <div className="bg-white rounded-lg p-4 border border-gray-200">
                        <h4 className="font-semibold text-gray-900 mb-2">Detailed Analysis</h4>
                        <p className="text-gray-700 leading-relaxed">{activity.details}</p>
                      </div>

                      {/* Recommendations */}
                      <div className={`mt-4 p-4 rounded-lg ${
                        activity.severity === 'high' ? 'bg-red-100 border border-red-300' :
                        activity.severity === 'medium' ? 'bg-amber-100 border border-amber-300' :
                        'bg-green-100 border border-green-300'
                      }`}>
                        <h4 className={`font-semibold mb-2 ${
                          activity.severity === 'high' ? 'text-red-900' :
                          activity.severity === 'medium' ? 'text-amber-900' :
                          'text-green-900'
                        }`}>
                          Recommendation
                        </h4>
                        <p className={`text-sm ${
                          activity.severity === 'high' ? 'text-red-800' :
                          activity.severity === 'medium' ? 'text-amber-800' :
                          'text-green-800'
                        }`}>
                          {activity.severity === 'high' && "⚠️ High risk detected. Avoid transacting with this wallet. Consider reporting to relevant authorities if fraudulent activity is confirmed."}
                          {activity.severity === 'medium' && "⚡ Exercise caution. Monitor this wallet closely and verify the legitimacy before engaging in any transactions."}
                          {activity.severity === 'low' && "✓ Activity appears normal. Continue standard monitoring practices and maintain regular security protocols."}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Summary Stats */}
            <div className="mt-8 p-6 bg-gray-50 rounded-xl border border-gray-200">
              <h3 className="font-semibold text-gray-900 mb-4">Analysis Summary</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="text-2xl font-bold text-red-600">
                    {activities.filter(a => a.severity === 'high').length}
                  </div>
                  <div className="text-sm text-gray-600">High Severity</div>
                </div>
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="text-2xl font-bold text-amber-600">
                    {activities.filter(a => a.severity === 'medium').length}
                  </div>
                  <div className="text-sm text-gray-600">Medium Severity</div>
                </div>
                <div className="bg-white rounded-lg p-4 border border-gray-200">
                  <div className="text-2xl font-bold text-green-600">
                    {activities.filter(a => a.severity === 'low').length}
                  </div>
                  <div className="text-sm text-gray-600">Low Severity</div>
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 px-8 py-4 flex justify-end">
            <button
              onClick={onClose}
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
