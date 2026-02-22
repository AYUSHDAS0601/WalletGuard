import { Shield, AlertTriangle } from "lucide-react";

interface RiskScoreProps {
  score: number;
}

export function RiskScore({ score }: RiskScoreProps) {
  const getRiskLevel = () => {
    if (score >= 75) return { label: "High Risk", color: "red", bgColor: "bg-red-500" };
    if (score >= 40) return { label: "Medium Risk", color: "amber", bgColor: "bg-amber-500" };
    return { label: "Low Risk", color: "green", bgColor: "bg-green-500" };
  };

  const risk = getRiskLevel();

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <Shield className="text-blue-600" size={24} />
        Risk Assessment
      </h2>

      <div className="flex items-center gap-6">
        {/* Risk Score Circle */}
        <div className="relative w-32 h-32">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke="#e5e7eb"
              strokeWidth="12"
              fill="none"
            />
            <circle
              cx="64"
              cy="64"
              r="56"
              stroke={`var(--color-${risk.color}-500)`}
              strokeWidth="12"
              fill="none"
              strokeDasharray={`${(score / 100) * 351.86} 351.86`}
              strokeLinecap="round"
              className={`transition-all duration-1000 ${
                risk.color === 'red' ? 'stroke-red-500' :
                risk.color === 'amber' ? 'stroke-amber-500' :
                'stroke-green-500'
              }`}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-900">{score}</div>
              <div className="text-xs text-gray-600">/ 100</div>
            </div>
          </div>
        </div>

        {/* Risk Details */}
        <div className="flex-1">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full mb-3 ${
            risk.color === 'red' ? 'bg-red-100 text-red-800' :
            risk.color === 'amber' ? 'bg-amber-100 text-amber-800' :
            'bg-green-100 text-green-800'
          }`}>
            {risk.color === 'green' ? (
              <Shield size={16} />
            ) : (
              <AlertTriangle size={16} />
            )}
            <span className="font-semibold">{risk.label}</span>
          </div>

          <p className="text-gray-600 text-sm mb-2">
            {score >= 75 && "This wallet exhibits multiple high-risk behaviors and should be approached with extreme caution."}
            {score >= 40 && score < 75 && "This wallet shows some suspicious patterns. Monitor activity closely."}
            {score < 40 && "This wallet appears to have normal transaction patterns with minimal risk indicators."}
          </p>

          <div className="flex flex-wrap gap-2 mt-3">
            {score >= 75 && (
              <>
                <span className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded">High Frequency</span>
                <span className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded">Mixer Usage</span>
              </>
            )}
            {score >= 40 && score < 75 && (
              <>
                <span className="text-xs px-2 py-1 bg-amber-100 text-amber-700 rounded">Unusual Patterns</span>
                <span className="text-xs px-2 py-1 bg-amber-100 text-amber-700 rounded">Large Transfers</span>
              </>
            )}
            {score < 40 && (
              <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">Normal Activity</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
