"use client"

import { useEffect, useState } from "react"
import { apiCall } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { Button } from "@/components/ui/button"
import { User, Bitcoin, Ethereum, Calendar, Globe, Gift, CheckCircle, XCircle, Copy } from "lucide-react"

interface UserProfile {
  name: string
  email?: string
  status: string
  usd_balance: number
  bitcoin_balance?: number
  ethereum_balance?: number
  bitcoin_balance_usd?: number
  ethereum_balance_usd?: number
  total_balance_usd?: number
  bitcoin_wallet?: string
  ethereum_wallet?: string
  referral_code?: string
  email_verified: boolean
  is_admin: boolean
  is_agent: boolean
  birthday_day?: number
  birthday_month?: number
  birthday_year?: number
  gender?: string
  user_country_code?: string
  zip_code?: string
  created_at: string
  referred_users_count?: number
}

export default function ProfileSettingsCard({ onReturnToDashboard }: { onReturnToDashboard: () => void }) {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()

  const fetchProfile = async () => {
    try {
      setIsLoading(true)
      const data = await apiCall<UserProfile>("/api/user/profile", "GET", null, true)
      setUser(data)
    } catch (err: any) {
      toast({
        title: "Error",
        description: err.message || "Failed to load profile",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchProfile()
  }, [])

  const copyReferralCode = async () => {
    if (user?.referral_code) {
      try {
        await navigator.clipboard.writeText(user.referral_code)
        toast({
          title: "Copied!",
          description: "Referral code copied to clipboard",
        })
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to copy referral code",
          variant: "destructive",
        })
      }
    }
  }

  const formatCrypto = (val?: number) => (val ?? 0).toFixed(6)
  const formatUSD = (val?: number) => (val ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })

  if (isLoading) return <p className="text-center py-20 text-gray-500">Loading profile...</p>
  if (!user)
    return (
      <p className="text-center py-20 text-red-500">
        Failed to load profile
        <Button onClick={onReturnToDashboard} className="ml-2" variant="outline">
          Return
        </Button>
      </p>
    )

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 mb-1">Profile Settings</h1>
          <Button onClick={onReturnToDashboard} variant="outline" className="bg-white">
            Back to Dashboard
          </Button>
        </div>

        {/* Portfolio Card */}
        <div className="bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] text-white rounded-2xl p-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-4">
              <div className="w-16 h-16 bg-white bg-opacity-20 rounded-2xl flex items-center justify-center">
                <User className="h-8 w-8 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold truncate">{user.name}</h2>
                {user.email && <p className="text-white/80 truncate">{user.email}</p>}
                <div className="flex items-center space-x-2 mt-2 flex-wrap">
                  <span className="bg-white bg-opacity-20 text-white text-xs px-2 py-1 rounded-full font-medium capitalize">{user.status}</span>
                  {user.email_verified && (
                    <span className="bg-green-500 bg-opacity-80 text-white text-xs px-2 py-1 rounded-full font-medium flex items-center space-x-1">
                      <CheckCircle className="h-3 w-3" />
                      <span>Verified</span>
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-2 text-right min-w-[140px]">
              <div>
                <p className="text-xs text-white/70">Total Portfolio</p>
                <p className="text-xl font-bold">${formatUSD(user.total_balance_usd)}</p>
              </div>
              <div>
                <p className="text-xs text-white/70">BTC</p>
                <p className="text-white font-bold">{formatCrypto(user.bitcoin_balance)} (~${formatUSD(user.bitcoin_balance_usd)})</p>
              </div>
              <div>
                <p className="text-xs text-white/70">ETH</p>
                <p className="text-white font-bold">{formatCrypto(user.ethereum_balance)} (~${formatUSD(user.ethereum_balance_usd)})</p>
              </div>
            </div>
          </div>
        </div>

        {/* Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Personal Info */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <User className="h-5 w-5 text-gray-600" />
              <h3 className="text-lg font-bold text-gray-900">Personal Information</h3>
            </div>
            <div className="space-y-2">
              {user.name && <p><span className="font-medium">Name:</span> {user.name}</p>}
              {user.gender && <p><span className="font-medium">Gender:</span> {user.gender}</p>}
              {user.birthday_day && user.birthday_month && (
                <p><Calendar className="inline h-4 w-4 mr-1 text-gray-500" />{user.birthday_day}/{user.birthday_month}{user.birthday_year ? `/${user.birthday_year}` : ""}</p>
              )}
              {user.user_country_code && <p><Globe className="inline h-4 w-4 mr-1 text-gray-500" />{user.user_country_code}</p>}
              {user.zip_code && <p><span className="font-medium">ZIP:</span> {user.zip_code}</p>}
            </div>
          </div>

          {/* Referral & Status */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <Gift className="h-5 w-5 text-purple-600" />
              <h3 className="text-lg font-bold text-gray-900">Referral & Account</h3>
            </div>
            <div className="space-y-2">
              {user.referral_code && (
                <div className="flex items-center space-x-2">
                  <code className="bg-gray-100 px-2 py-1 rounded font-mono">{user.referral_code}</code>
                  <Button size="sm" variant="outline" onClick={copyReferralCode}>
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              )}
              <p><span className="font-medium">Referred Users:</span> {user.referred_users_count ?? 0}</p>
              <p><span className="font-medium">Admin:</span> {user.is_admin ? "Yes" : "No"}</p>
              <p><span className="font-medium">Agent:</span> {user.is_agent ? "Active" : "Inactive"}</p>
            </div>
          </div>

          {/* Wallet Info */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <Bitcoin className="h-5 w-5 text-yellow-500" />
              <h3 className="text-lg font-bold text-gray-900">Wallets</h3>
            </div>
            <div className="space-y-2">
              {user.bitcoin_wallet && <p><span className="font-medium">BTC Wallet:</span> {user.bitcoin_wallet}</p>}
              {user.ethereum_wallet && <p><span className="font-medium">ETH Wallet:</span> {user.ethereum_wallet}</p>}
            </div>
          </div>

          {/* Account Timeline */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <Calendar className="h-5 w-5 text-orange-500" />
              <h3 className="text-lg font-bold text-gray-900">Account Timeline</h3>
            </div>
            <div className="space-y-2">
              <p><span className="font-medium">Member Since:</span> {new Date(user.created_at).toLocaleDateString()}</p>
              <p><span className="font-medium">Account Age:</span> {Math.floor((Date.now() - new Date(user.created_at).getTime()) / (1000*60*60*24))} days</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
