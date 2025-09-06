"use client"

import { useEffect, useState } from "react"
import { apiCall } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { Button } from "@/components/ui/button"
import { User, Mail, Shield, Star, Gift, Calendar, CheckCircle, XCircle, Copy, ArrowLeft } from "lucide-react"

interface UserProfile {
  email: string
  name?: string
  status: string
  points_balance: number
  referral_code?: string
  email_verified: boolean
  is_admin: boolean
  is_agent: boolean
  created_at: string
}

export default function ProfileSection({ onReturnToDashboard }: { onReturnToDashboard: () => void }) {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    const loadProfile = async () => {
      setIsLoading(true)
      try {
        const data = await apiCall<UserProfile>("/users/me", "GET", null, true) // Updated API call
        setUser(data)
      } catch (error: any) {
        toast({
          title: "Error",
          description: error.message || "Failed to load profile.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }
    loadProfile()
  }, [toast])

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

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
            <div className="flex items-center justify-center min-h-[300px]">
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <User className="h-6 w-6 text-purple-600 animate-pulse" />
                </div>
                <p className="text-gray-500 text-lg">Loading your profile...</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
            <div className="flex items-center justify-center min-h-[300px]">
              <div className="text-center">
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <XCircle className="h-6 w-6 text-red-600" />
                </div>
                <p className="text-red-500 text-lg mb-4">Failed to load profile</p>
                <Button onClick={onReturnToDashboard} variant="outline" className="bg-white">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Return to Dashboard
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-1">Profile Settings</h1>
            <p className="text-gray-600">Manage your account information and preferences</p>
          </div>
          <Button onClick={onReturnToDashboard} variant="outline" className="bg-white">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>

        {/* Profile Overview Card */}
        <div className="bg-gradient-to-r from-purple-600 to-pink-500 rounded-2xl p-6 text-white">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center space-x-4 min-w-0 flex-1">
              <div className="w-16 h-16 bg-white bg-opacity-20 rounded-2xl flex items-center justify-center flex-shrink-0">
                <User className="h-8 w-8 text-white" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 className="text-2xl font-bold truncate">{user.name || "User"}</h2>
                <p className="text-purple-100 truncate">{user.email}</p>
                <div className="flex items-center space-x-2 mt-2 flex-wrap">
                  <span className="bg-white bg-opacity-20 text-white text-xs px-2 py-1 rounded-full font-medium capitalize">
                    {user.status}
                  </span>
                  {user.email_verified && (
                    <span className="bg-green-500 bg-opacity-80 text-white text-xs px-2 py-1 rounded-full font-medium flex items-center space-x-1">
                      <CheckCircle className="h-3 w-3" />
                      <span>Verified</span>
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="text-right flex-shrink-0 min-w-0 w-auto max-w-[140px] sm:max-w-[200px]">
              <p className="text-purple-200 text-xs sm:text-sm whitespace-nowrap">Points Balance</p>
              <p className="text-xl sm:text-3xl font-bold break-all leading-tight overflow-hidden">
                {user.points_balance.toLocaleString()}
              </p>
              <p className="text-purple-200 text-xs sm:text-sm whitespace-nowrap">points</p>
            </div>
          </div>
        </div>

        {/* Account Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Contact Information */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center">
                <Mail className="h-5 w-5 text-blue-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Contact Information</h3>
            </div>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Email Address</p>
                <p className="text-gray-900 font-medium">{user.email}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Full Name</p>
                <p className="text-gray-900 font-medium">{user.name || "Not provided"}</p>
              </div>
              <div className="flex items-center space-x-2">
                <p className="text-sm text-gray-500">Email Status:</p>
                {user.email_verified ? (
                  <span className="flex items-center space-x-1 text-green-600 text-sm font-medium">
                    <CheckCircle className="h-4 w-4" />
                    <span>Verified</span>
                  </span>
                ) : (
                  <span className="flex items-center space-x-1 text-orange-600 text-sm font-medium">
                    <XCircle className="h-4 w-4" />
                    <span>Unverified</span>
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Account Status */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center">
                <Shield className="h-5 w-5 text-green-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Account Status</h3>
            </div>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Account Type</p>
                <p className="text-gray-900 font-medium capitalize">{user.status}</p>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Admin Access:</span>
                <span className={`text-sm font-medium ${user.is_admin ? "text-green-600" : "text-gray-400"}`}>
                  {user.is_admin ? "Yes" : "No"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500">Agent Status:</span>
                <span className={`text-sm font-medium ${user.is_agent ? "text-blue-600" : "text-gray-400"}`}>
                  {user.is_agent ? "Active" : "Inactive"}
                </span>
              </div>
            </div>
          </div>

          {/* Referral Information */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-purple-100 rounded-xl flex items-center justify-center">
                <Gift className="h-5 w-5 text-purple-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Referral Program</h3>
            </div>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 mb-2">Your Referral Code</p>
                {user.referral_code ? (
                  <div className="flex items-center space-x-2">
                    <code className="bg-gray-100 px-3 py-2 rounded-lg text-gray-900 font-mono text-sm flex-1">
                      {user.referral_code}
                    </code>
                    <Button size="sm" variant="outline" onClick={copyReferralCode} className="bg-white">
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">No referral code assigned</p>
                )}
              </div>
              <p className="text-xs text-gray-500">
                Share your referral code with friends to earn bonus points when they join!
              </p>
            </div>
          </div>

          {/* Account Timeline */}
          <div className="bg-white rounded-2xl p-6 border border-gray-100">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-orange-100 rounded-xl flex items-center justify-center">
                <Calendar className="h-5 w-5 text-orange-600" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Account Timeline</h3>
            </div>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Member Since</p>
                <p className="text-gray-900 font-medium">
                  {new Date(user.created_at).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Account Age</p>
                <p className="text-gray-900 font-medium">
                  {Math.floor((Date.now() - new Date(user.created_at).getTime()) / (1000 * 60 * 60 * 24))} days
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Points Summary Card */}
        <div className="bg-white rounded-2xl p-6 border border-gray-100 border-2 border-dashed border-purple-200">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-pink-100 rounded-xl flex items-center justify-center">
              <Star className="h-5 w-5 text-pink-600" />
            </div>
            <h3 className="text-lg font-bold text-gray-900">Points Summary</h3>
            <span className="bg-pink-100 text-pink-700 text-xs px-2 py-1 rounded-full font-medium">
              Current Balance
            </span>
          </div>
          <div className="text-center py-4">
            <p className="text-4xl font-bold text-gray-900 mb-2 break-all px-4">
              {user.points_balance.toLocaleString()}
            </p>
            <p className="text-gray-500">points available for redemption</p>
          </div>
        </div>
      </div>
    </div>
  )
      }
