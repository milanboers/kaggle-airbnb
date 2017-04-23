import csv
import numpy as np
import pandas
from datetime import date
from agebuckets import getbucket
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import scale

def realage(age):
	return -1 if age == '' or float(age) > 110 or float(age) < 5 else float(age)

def scaled(ds):
	d = (np.max(ds) - np.min(ds)) / 2
	return (ds - np.min(ds)) / np.float32(d) - 1

def scaledWmm(ds, mini, maxi):
	d = (maxi - mini) / 2.0
	return (ds - mini) / np.float32(d) - 1

# Scaled all values except NaT, makes NaT the average value
def scaledWithNaT(ds):
	filtered = ds[ds > 0]
	d = (np.max(filtered) - np.min(filtered)) / 2.0
	n = (ds - np.min(filtered)) / np.float32(d) - 1
	filteredn = n[ds > 0]
	return np.where(ds < 0, np.average(filteredn).repeat(ds.size), n)

def scaleCat(ds):
	return scaled(np.array(ds).astype(np.float32))

def inputcreate(filename):
	users = []

	# Read training file
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next()
		for row in reader:
			users.append(row)

	users = np.array(users)

	# Create input matrix
	numusers = np.empty((users.shape[0], 969))
	starti = 0

	# Remove id column
	ids = users[:,0]
	users = users[:,1:]
	# Account created to timestamp
	times = pandas.to_datetime(users[:,0], format='%Y-%m-%d')
	numusers[:,starti] = scaledWmm(times.astype(np.int64), 1262304000000000000, 1412035200000000000)
	starti += 1
	# First active
	firstactivetimes = pandas.to_datetime(users[:,1], format='%Y%m%d%H%M%S')
	numusers[:,starti] = scaledWmm(firstactivetimes.astype(np.int64), 1237437175000000000, 1412121541000000000)
	starti += 1
	# TODO: kijken of dit beter kan
	# Date first booking
	#dfb = pandas.to_datetime(users[:,2], format='%Y-%m-%d', errors='coerce').astype(np.int64)
	#numusers[:,2] = scaledWithNaT(dfb)
	# Date first booking (1 if date, 0 otherwise)
	#numusers[:,3] = scaleCat(dfb >= 0)
	# Gender
	gender = users[:,3]
	# Unknown
	numusers[:,starti] = scaleCat(gender == '-unknown-')
	starti += 1
	# Male
	numusers[:,starti] = scaleCat(gender == 'MALE')
	starti += 1
	# Female
	numusers[:,starti] = scaleCat(gender == 'FEMALE')
	starti += 1
	# Other
	numusers[:,starti] = scaleCat(gender == 'OTHER')
	starti += 1
	# Age
	ages = users[:,4]
	ages = np.array([realage(x) for x in ages])
	numusers[:,starti] = scaledWithNaT(ages)
	starti += 1
	# Age given
	numusers[:,starti] = scaleCat(ages != -1)
	starti += 1
	# Signup method = facebook
	signupmethod = users[:,5]
	numusers[:,starti] = scaleCat(signupmethod == 'facebook')
	starti += 1
	# Signup method = basic
	numusers[:,starti] = scaleCat(signupmethod == 'basic')
	starti += 1
	# Signup method = google
	numusers[:,starti] = scaleCat(signupmethod == 'google')
	starti += 1
	# Signup flows
	signupflows = users[:,6]
	distinctflows = ['0', '8', '12', '21', '23', '25']
	for flow in distinctflows:
		numusers[:,starti] = scaleCat(signupflows == flow)
		starti += 1
	# Languages
	languages = users[:,7]
	distinctlanguages = ['ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'hu', 'id', 'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'zh']
	for language in distinctlanguages:
		numusers[:,starti] = scaleCat(languages == language)
		starti += 1
	# Affiliate channel
	affiliatechannels = users[:,8]
	distinctchannels = ['content', 'direct', 'other', 'remarketing', 'sem-brand', 'sem-non-brand', 'seo']
	for channel in distinctchannels:
		numusers[:,starti] = scaleCat(affiliatechannels == channel)
		starti += 1
	# Affiliate provider
	affiliateproviders = users[:,9]
	distinctproviders = ['baidu', 'bing', 'craigslist', 'daum', 'direct', 'email-marketing', 'facebook', 'facebook-open-graph', 'google', 'gsp', 'meetup', 'naver', 'other', 'padmapper', 'vast', 'yahoo', 'yandex']
	for provider in distinctproviders:
		numusers[:,starti] = scaleCat(affiliateproviders == provider)
		starti += 1
	# First affiliate tracked
	firsttracked = users[:,10]
	distincttracked = ['', 'linked', 'local ops', 'marketing', 'omg', 'product', 'tracked-other', 'untracked']
	for track in distincttracked:
		numusers[:,starti] = scaleCat(firsttracked == track)
		starti += 1
	# Signup app
	signupapp = users[:,11]
	distinctsingapp = ['Android', 'Moweb', 'Web', 'iOS']
	for app in distinctsingapp:
		numusers[:,starti] = scaleCat(signupapp == app)
		starti += 1
	# First device type
	firstdevicetype = users[:,12]
	distinctfdt = ['Android Phone', 'Android Tablet', 'Desktop (Other)', 'Mac Desktop', 'Other/Unknown', 'SmartPhone (Other)', 'Windows Desktop', 'iPad', 'iPhone']
	for fdt in distinctfdt:
		numusers[:,starti] = scaleCat(firstdevicetype == fdt)
		starti += 1
	# First browser
	firstbrowser = users[:,13]
	distinctfirstbrowser = ['-unknown-', 'AOL Explorer', 'Android Browser', 'Apple Mail', 'BlackBerry Browser', 'Chrome', 'Chrome Mobile', 'Chromium', 'CometBird', 'IE', 'IE Mobile', 'Safari', 'SeaMonkey', 'Silk', 'SiteKiosk', 'Sogou Explorer', 'Yandex.Browser', 'wOSBrowser']
	for fb in distinctfirstbrowser:
		numusers[:,starti] = scaleCat(firstbrowser == fb)
		starti += 1
	# Add age/gender bucket features
	bucketdata = []
	for user in users:
		a = realage(user[4])
		g = user[3]
		bucketdata.append(getbucket(a, g))
	numusers[:,starti:starti+14] = np.array(bucketdata)
	starti += 14
	# Day of week for account created
	weekdaydata = []
	for time in times:
		weekdaydata.append([1 if i == time.weekday() else -1 for i in range(7)])
	numusers[:,starti:starti+7] = np.array(weekdaydata)
	starti += 7
	# Month for account created
	monthdata = []
	for time in times:
		monthdata.append([1 if i == time.month else -1 for i in range(12)])
	numusers[:,starti:starti+12] = np.array(monthdata)
	starti += 12
	# Day of week for first active
	actstamps = pandas.to_datetime(users[:,1])
	actweekday = []
	for time in actstamps:
		actweekday.append([1 if i == time.weekday() else -1 for i in range(7)])
	numusers[:,starti:starti+7] = np.array(actweekday)
	starti += 7
	# Month for first active
	actmonth = []
	for time in actstamps:
		actmonth.append([1 if i == time.month else -1 for i in range(12)])
	numusers[:,starti:starti+12] = np.array(actmonth)
	starti += 12
	
	langmap = {
		'eng': 'en',
		'deu': 'de',
		'spa': 'es',
		'fra': 'fr',
		'ita': 'it',
		'nld': 'nl',
		'por': 'pt',
	}
	
	countriesbylang = dict()
	# Countries.csv
	with open('countries.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next()
		for row in reader:
			countriesbylang[langmap[row[5]]] = row
	
	# Languages
	ls = set(['de','es','fr','it','nl','pt','en'])
	# Distance to native language country
	distance_kms = []
	# Size of native language country
	distance_size = []
	# Native language country info available?
	distance_avb = []
	distance_lev = []
	for language in languages:
		if language in ls:
			distance = float(countriesbylang[language][3])
			size = float(countriesbylang[language][4])
			levenshtein = float(countriesbylang[language][6])
			distance_kms.append(distance)
			distance_size.append(size)
			distance_lev.append(levenshtein)
			distance_avb.append(1)
		else:
			# Average distance
			distance_kms.append(6687.0631)
			# Average size
			distance_size.append(1681120.143)
			distance_lev.append(72.14142857)
			distance_avb.append(-1)
	numusers[:,starti] = scaledWmm(np.array(distance_kms), 0, 8636.631)
	numusers[:,starti+1] = scaledWmm(np.array(distance_size), 41543, 9826675)
	numusers[:,starti+2] = scaledWmm(np.array(distance_lev), 0, 95.45)
	numusers[:,starti+3] = np.array(distance_avb)
	starti += 4
	
	keyswithsessions = set()
	#keyswithbrs = set()
	sessions = dict()
	# Read sessions file
	print "Reading session file"
	i = 0
	with open('sessions.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		reader.next()
		for row in reader:
			if not row[0] in sessions:
				keyswithsessions.add(row[0])
				sessions[row[0]] = []
			#if row[2] == 'booking_request':
			#	keyswithbrs.add(row[0])
			duo1.add((row[2], row[3]))
			sessions[row[0]].append(row)
			i += 1
	print i, "sessions"
	
	# Add sessions to numusers
	distincttriads = [('', 'message_post', 'message_post'), ('10', 'message_post', 'message_post'), ('11', 'message_post', 'message_post'), ('12', 'message_post', 'message_post'), ('15', 'message_post', 'message_post'), ('about_us', '-unknown-', '-unknown-'), ('accept_decline', 'view', 'host_respond_page'), ('account', '-unknown-', '-unknown-'), ('acculynk_bin_check_failed', '-unknown-', '-unknown-'), ('acculynk_bin_check_success', '-unknown-', '-unknown-'), ('acculynk_load_pin_pad', '-unknown-', '-unknown-'), ('acculynk_pin_pad_error', '-unknown-', '-unknown-'), ('acculynk_pin_pad_inactive', '-unknown-', '-unknown-'), ('acculynk_pin_pad_success', '-unknown-', '-unknown-'), ('acculynk_session_obtained', '-unknown-', '-unknown-'), ('active', '-unknown-', '-unknown-'), ('add_business_address_colorbox', '-unknown-', '-unknown-'), ('add_guest_colorbox', '-unknown-', '-unknown-'), ('add_guests', '-unknown-', '-unknown-'), ('add_note', 'submit', 'wishlist_note'), ('agree_terms_check', '-unknown-', '-unknown-'), ('agree_terms_uncheck', '-unknown-', '-unknown-'), ('airbnb_picks', 'view', 'airbnb_picks_wishlists'), ('airbrb', '-unknown-', '-unknown-'), ('ajax_check_dates', 'click', 'change_contact_host_dates'), ('ajax_get_referrals_amt', '-unknown-', '-unknown-'), ('ajax_get_results', 'click', 'view_search_results'), ('ajax_google_translate', '-unknown-', '-unknown-'), ('ajax_google_translate_description', '-unknown-', '-unknown-'), ('ajax_google_translate_reviews', 'click', 'translate_listing_reviews'), ('ajax_image_upload', '-unknown-', '-unknown-'), ('ajax_ldp', '-unknown-', '-unknown-'), ('ajax_lwlb_contact', 'click', 'contact_host'), ('ajax_payout_edit', '-unknown-', '-unknown-'), ('ajax_payout_options_by_country', '-unknown-', '-unknown-'), ('ajax_payout_split_edit', '-unknown-', '-unknown-'), ('ajax_photo_widget', '-unknown-', '-unknown-'), ('ajax_photo_widget_form_iframe', '-unknown-','-unknown-'), ('ajax_price_and_availability', 'click', 'alteration_field'), ('ajax_referral_banner_experiment_type', '-unknown-', '-unknown-'), ('ajax_referral_banner_type', '-unknown-', '-unknown-'), ('ajax_refresh_subtotal', 'click', 'change_trip_characteristics'), ('ajax_send_message', '-unknown-', '-unknown-'), ('ajax_special_offer_dates_available', 'click', 'special_offer_field'), ('ajax_statsd', '-unknown-', '-unknown-'), ('ajax_worth', 'submit', 'calculate_worth'), ('apply', '-unknown-', '-unknown-'), ('apply_code', '-unknown-','-unknown-'), ('apply_coupon_click', 'click', 'apply_coupon_click'), ('apply_coupon_click_success', 'click', 'apply_coupon_click_success'), ('apply_coupon_error', 'click', 'apply_coupon_error'), ('apply_coupon_error_type', '-unknown-', '-unknown-'), ('apply_reservation', 'submit', 'apply_coupon'), ('approve', '-unknown-', '-unknown-'), ('approve', 'submit', 'host_respond'), ('ask_question', 'submit', 'contact_host'), ('at_checkpoint', 'booking_request', 'at_checkpoint'), ('authenticate', 'submit', 'login'), ('authenticate', 'view', 'login_page'), ('authorize', '-unknown-', '-unknown-'), ('available', '-unknown-', '-unknown-'), ('available', 'data', 'trip_availability'), ('badge', '-unknown-', '-unknown-'), ('become_user', '-unknown-', '-unknown-'), ('book', 'view', 'p4'), ('booking', 'booking_response', 'booking'), ('braintree_client_token', '', ''), ('business_travel', '-unknown-', '-unknown-'), ('calendar_tab_inner2', '-unknown-', '-unknown-'), ('callback', 'partner_callback', 'oauth_response'), ('campaigns', '', ''), ('campaigns', '-unknown-', '-unknown-'), ('cancel', 'submit', 'guest_cancellation'), ('cancellation_policies', 'view', 'cancellation_policies'), ('cancellation_policy_click', 'click', 'cancellation_policy_click'), ('change', 'view', 'change_or_alter'), ('change_availability', '-unknown-', '-unknown-'), ('change_availability', 'submit', 'change_availability'), ('change_currency', '-unknown-', '-unknown-'), ('change_default_payout', '-unknown-', '-unknown-'), ('change_password', 'submit', 'change_password'), ('check', '', ''), ('city_count', '-unknown-', '-unknown-'), ('clear_reservation', '-unknown-', '-unknown-'), ('click', '-unknown-', '-unknown-'), ('click', 'click', 'book_it'), ('click', 'click', 'cancellation_policy'), ('click', 'click', 'click_about_host'), ('click', 'click', 'click_amenities'), ('click', 'click', 'click_reviews'), ('click', 'click', 'complete_booking'), ('click', 'click', 'contact_host'), ('click', 'click', 'instant_book'), ('click', 'click', 'move_map'), ('click', 'click', 'photos'), ('click', 'click', 'request_to_book'), ('click', 'click', 'share'), ('clickthrough', '-unknown-', '-unknown-'), ('collections', '-unknown-', '-unknown-'), ('collections', 'view', 'user_wishlists'), ('complete', '-unknown-', '-unknown-'), ('complete_redirect', '-unknown-', '-unknown-'), ('complete_status', '-unknown-', '-unknown-'), ('concierge', '-unknown-', '-unknown-'), ('confirm_email', '-unknown-', '-unknown-'), ('confirm_email', 'click', 'confirm_email'), ('confirm_email', 'click', 'confirm_email_link'), ('confirmation', '-unknown-', '-unknown-'), ('connect', '-unknown-', '-unknown-'), ('connect', 'submit', 'oauth_login'), ('contact_new', '-unknown-', '-unknown-'), ('countries', '-unknown-', '-unknown-'), ('country_options', '-unknown-', '-unknown-'), ('coupon_code_click', 'click', 'coupon_code_click'), ('coupon_field_focus', 'click', 'coupon_field_focus'), ('create', '-unknown-', '-unknown-'), ('create', 'submit', 'create_alteration_request'), ('create', 'submit', 'create_listing'), ('create', 'submit', 'create_payment_instrument'), ('create', 'submit', 'create_phone_numbers'), ('create', 'submit', 'create_user'), ('create', 'submit', 'signup'), ('create', 'view', 'list_your_space'), ('create_ach', '-unknown-', '-unknown-'), ('create_airbnb', '-unknown-', '-unknown-'), ('create_multiple', '-unknown-', '-unknown-'), ('create_paypal', '-unknown-', '-unknown-'), ('currencies', '', ''), ('currencies', '-unknown-', '-unknown-'), ('custom_recommended_destinations', '-unknown-', '-unknown-'), ('dashboard', 'view', 'dashboard'), ('deactivate', '-unknown-', '-unknown-'), ('deactivated', 'view', 'host_standard_suspension'), ('deauthorize', '-unknown-', '-unknown-'), ('decision_tree', '-unknown-', '-unknown-'), ('delete', '-unknown-', '-unknown-'), ('delete', '-unknown-', 'phone_numbers'), ('delete', '-unknown-', 'reservations'), ('delete', 'submit', 'deactivate_user_account'), ('delete', 'submit', 'delete_listing'), ('delete', 'submit', 'delete_listing_description'), ('delete', 'submit', 'delete_phone_numbers'), ('department', '-unknown-', '-unknown-'), ('departments', '-unknown-', '-unknown-'), ('desks', '-unknown-', '-unknown-'), ('destroy', '-unknown-', '-unknown-'), ('destroy', 'submit', 'delete_payment_instrument'), ('detect_fb_session', '-unknown-', '-unknown-'), ('disaster_action', '', ''), ('domains', '-unknown-', '-unknown-'), ('edit', '-unknown-', '-unknown-'), ('edit', 'view', 'edit_profile'), ('edit_verification', 'view', 'profile_verifications'), ('email_by_key', '-unknown-', '-unknown-'), ('email_itinerary_colorbox', '-unknown-', '-unknown-'), ('email_share', 'submit', 'email_wishlist'), ('email_wishlist', 'click', 'email_wishlist_button'), ('endpoint_error', '-unknown-', '-unknown-'), ('envoy_bank_details_redirect', '-unknown-', '-unknown-'), ('envoy_form','-unknown-', '-unknown-'), ('events', '-unknown-', '-unknown-'), ('facebook_auto_login', '-unknown-', '-unknown-'), ('faq', '-unknown-', '-unknown-'), ('faq_category', '-unknown-', '-unknown-'), ('faq_experiment_ids', '-unknown-', '-unknown-'), ('feed', '-unknown-', '-unknown-'), ('forgot_password', 'click', 'forgot_password'), ('forgot_password', 'submit', 'forgot_password'), ('founders', '-unknown-', '-unknown-'), ('friend_listing', '-unknown-', '-unknown-'), ('friends', 'view', 'friends_wishlists'), ('friends_new', '-unknown-', '-unknown-'), ('glob', '-unknown-', '-unknown-'), ('google_importer', '-unknown-', '-unknown-'), ('guarantee', 'view', 'host_guarantee'), ('guest_billing_receipt', '-unknown-', '-unknown-'), ('guest_booked_elsewhere', 'message_post', 'message_post'), ('handle_vanity_url', '-unknown-', '-unknown-'), ('hard_fallback_submit', '-unknown-', '-unknown-'), ('has_profile_pic', '-unknown-', '-unknown-'), ('header_userpic', 'data', 'header_userpic'), ('home_safety_landing', '-unknown-', '-unknown-'), ('home_safety_terms', '-unknown-', '-unknown-'), ('hospitality', '-unknown-', '-unknown-'), ('hospitality_standards', '-unknown-', '-unknown-'), ('host_2013', '-unknown-', '-unknown-'), ('host_cancel', '-unknown-', '-unknown-'), ('host_summary', '-unknown-', '-unknown-'), ('host_summary', 'view', 'host_home'), ('hosting_social_proof', '-unknown-', '-unknown-'), ('how_it_works', '-unknown-', '-unknown-'), ('identity', '-unknown-', '-unknown-'), ('image_order', '-unknown-', '-unknown-'), ('impressions', 'view', 'p4'), ('index', '', ''), ('index', '-unknown-', '-unknown-'), ('index', 'data', 'reservations'), ('index', 'data', 'user_tax_forms'), ('index', 'view', 'account_payment_methods'), ('index', 'view', 'homepage'), ('index', 'view', 'listing_descriptions'), ('index', 'view', 'message_inbox'), ('index', 'view', 'message_thread'), ('index', 'view', 'user_wishlists'), ('index', 'view', 'view_ghosting_reasons'), ('index', 'view', 'view_ghostings'), ('index', 'view', 'view_identity_verifications'), ('index', 'view', 'view_locations'), ('index', 'view', 'view_reservations'), ('index', 'view', 'view_resolutions'), ('index', 'view', 'view_search_results'), ('index', 'view', 'view_user_real_names'), ('index', 'view', 'your_listings'), ('invalid_action', '-unknown-', '-unknown-'), ('issue', '-unknown-', '-unknown-'), ('itinerary', 'view', 'guest_itinerary'), ('jumio', '-unknown-', '-unknown-'), ('jumio_redirect', '-unknown-', '-unknown-'), ('jumio_token', '-unknown-', '-unknown-'), ('kba', '-unknown-', '-unknown-'), ('kba_update', '-unknown-', '-unknown-'), ('languages_multiselect', '-unknown-', '-unknown-'), ('life', '-unknown-', '-unknown-'), ('listing','view', 'p3'), ('listings', '-unknown-', '-unknown-'), ('listings', 'view', 'user_listings'), ('load_more', '-unknown-', '-unknown-'), ('locale_from_host', '-unknown-', '-unknown-'), ('localization_settings', '', ''), ('localization_settings', '-unknown-', '-unknown-'),('localized', '-unknown-', '-unknown-'), ('locations', '-unknown-', '-unknown-'), ('login', '-unknown-', '-unknown-'), ('login', 'view', 'login_page'), ('login_modal', 'view', 'login_modal'), ('lookup', '', ''), ('manage_listing', 'view', 'manage_listing'), ('maybe_information', 'message_post', 'message_post'), ('media_resources', '-unknown-', '-unknown-'), ('message', '-unknown-', '-unknown-'), ('message_to_host_change', 'click', 'message_to_host_change'), ('message_to_host_focus', 'click', 'message_to_host_focus'), ('mobile_landing_page', '-unknown-', '-unknown-'), ('mobile_oauth_callback', '-unknown-', '-unknown-'), ('multi', '-unknown-', '-unknown-'), ('multi_message', 'message_post', 'message_post'), ('multi_message_attributes', '-unknown-', '-unknown-'), ('my', 'view', 'user_wishlists'), ('my_listings', 'view', 'your_reservations'), ('my_reservations', 'view', 'your_reservations'), ('new', '-unknown-', '-unknown-'), ('new', 'view', 'list_your_space'), ('new_host', '-unknown-', '-unknown-'), ('new_session', '-unknown-', '-unknown-'), ('notifications', '-unknown-', '-unknown-'), ('notifications', 'data', 'notifications'), ('notifications', 'submit', 'notifications'), ('notifications', 'view', 'account_notification_settings'), ('nyan', '-unknown-', '-unknown-'), ('office_location', '-unknown-', '-unknown-'), ('onenight', '-unknown-', '-unknown-'), ('open_graph_setting', '-unknown-', '-unknown-'), ('open_hard_fallback_modal', '-unknown-', '-unknown-'), ('other_hosting_reviews', '-unknown-', '-unknown-'), ('other_hosting_reviews_first', '-unknown-', '-unknown-'), ('overview', '-unknown-', '-unknown-'), ('p4_refund_policy_terms', 'click', 'p4_refund_policy_terms'), ('p4_terms', 'click', 'p4_terms'), ('patch', '-unknown-', '-unknown-'), ('patch', 'modify', 'modify_reservations'), ('patch', 'modify', 'modify_users'), ('pay', '-unknown-', '-unknown-'), ('payment_instruments', '-unknown-', '-unknown-'), ('payment_instruments', 'data', 'payment_instruments'), ('payment_methods', '-unknown-', '-unknown-'), ('payoneer_account_redirect', '-unknown-', '-unknown-'), ('payoneer_signup_complete', '-unknown-', '-unknown-'), ('payout_delete', '-unknown-', '-unknown-'), ('payout_preferences', 'view', 'account_payout_preferences'), ('payout_update', '-unknown-', '-unknown-'), ('pending', '-unknown-', '-unknown-'), ('pending', 'booking_request', 'pending'), ('pending_tickets', '-unknown-', '-unknown-'), ('personalize', 'data', 'wishlist_content_update'), ('phone_number_widget', '-unknown-', '-unknown-'), ('phone_verification', '', ''), ('phone_verification_call_taking_too_long', '-unknown-', '-unknown-'), ('phone_verification_error', '-unknown-', '-unknown-'), ('phone_verification_modal', '-unknown-', '-unknown-'), ('phone_verification_number_submitted_for_call', '-unknown-', '-unknown-'), ('phone_verification_number_submitted_for_sms', '-unknown-', '-unknown-'), ('phone_verification_number_sucessfully_submitted', '-unknown-', '-unknown-'), ('phone_verification_phone_number_removed', '-unknown-', '-unknown-'), ('phone_verification_success', 'click', 'phone_verification_success'), ('photography', '-unknown-', '-unknown-'), ('photography_update', '-unknown-', '-unknown-'), ('place_worth', 'view', 'place_worth'), ('plaxo_cb', '-unknown-', '-unknown-'), ('popular', 'view', 'popular_wishlists'), ('popular_listing', '-unknown-', '-unknown-'), ('populate_from_facebook', '-unknown-', '-unknown-'), ('populate_help_dropdown', '-unknown-', '-unknown-'), ('position', '-unknown-', '-unknown-'), ('preapproval', 'message_post', 'message_post'), ('press_content', '-unknown-', '-unknown-'), ('press_news', '-unknown-', '-unknown-'), ('press_release', '-unknown-', '-unknown-'), ('pricing', '-unknown-', '-unknown-'), ('print_confirmation', '-unknown-', '-unknown-'), ('privacy', 'view', 'account_privacy_settings'), ('profile_pic', '-unknown-', '-unknown-'), ('push_notification_callback', '-unknown-', '-unknown-'), ('qt2', 'view', 'message_thread'), ('qt_reply_v2', '-unknown-', '-unknown-'), ('qt_reply_v2', 'submit', 'send_message'), ('qt_with', 'data', 'lookup_message_thread'), ('questions', '-unknown-', '-unknown-'), ('rate', '-unknown-', '-unknown-'), ('reactivate', '-unknown-', '-unknown-'), ('read_policy_click', 'click', 'read_policy_click'), ('receipt', 'view', 'guest_receipt'), ('recent_reservations', '-unknown-', '-unknown-'), ('recommend', '-unknown-', '-unknown-'), ('recommendation_page', '-unknown-', '-unknown-'), ('recommendations', '-unknown-', '-unknown-'), ('recommendations', 'data', 'listing_recommendations'), ('recommendations', 'data', 'user_friend_recommendations'), ('recommended_listings', '-unknown-', '-unknown-'), ('redirect', '-unknown-', '-unknown-'), ('references', 'view', 'profile_references'), ('referrer_status', '-unknown-', '-unknown-'), ('refund_guest_cancellation', 'submit', 'host_refund_guest'), ('relationship', '-unknown-', '-unknown-'), ('remove_dashboard_alert', '-unknown-', '-unknown-'), ('remove_dashboard_alert', 'click', 'remove_dashboard_alert'), ('rentals', '-unknown-', '-unknown-'), ('report', '-unknown-', '-unknown-'), ('reputation', '-unknown-', '-unknown-'), ('request_new_confirm_email', 'click', 'request_new_confirm_email'), ('request_photography', '-unknown-', '-unknown-'), ('requested', 'submit', 'post_checkout_action'), ('requested', 'view', 'p5'), ('requirements', '-unknown-', '-unknown-'), ('reservation', '-unknown-', '-unknown-'), ('reset_calendar', '-unknown-', '-unknown-'), ('respond', 'submit', 'respond_to_alteration_request'), ('rest-of-world', '-unknown-', '-unknown-'), ('revert_to_admin', '-unknown-', '-unknown-'), ('review_page', '-unknown-', '-unknown-'), ('reviews', '-unknown-', '-unknown-'), ('reviews', 'data', 'listing_reviews'), ('reviews', 'data', 'user_reviews'), ('reviews', 'view', 'profile_reviews'), ('reviews_new', '-unknown-', '-unknown-'), ('salute', '-unknown-', '-unknown-'), ('sandy', '-unknown-', '-unknown-'), ('satisfy', '', ''), ('search', '-unknown-', '-unknown-'), ('search', 'click', 'view_search_results'), ('search_results', 'click', 'view_search_results'), ('set_default', '-unknown-', '-unknown-'), ('set_default', 'submit', 'set_default_payment_instrument'), ('set_minimum_payout_amount', '-unknown-', '-unknown-'), ('set_password', 'submit', 'set_password'), ('set_password', 'view', 'set_password_page'), ('set_user', 'submit', 'create_listing'), ('settings', '-unknown-', '-unknown-'), ('show', '', ''), ('show', '-unknown-', '-unknown-'), ('show', 'data', 'translations'), ('show', 'view', 'alteration_request'), ('show', 'view', 'p1'), ('show', 'view', 'p3'), ('show', 'view', 'user_profile'), ('show', 'view', 'view_identity_verifications'), ('show', 'view', 'view_listing'), ('show', 'view', 'view_security_checks'), ('show', 'view', 'wishlist'), ('show_code', '-unknown-', '-unknown-'), ('show_personalize', 'data', 'user_profile_content_update'), ('signature', '-unknown-', '-unknown-'), ('signed_out_modal', '', ''), ('signup_login', 'view', 'signup_login_page'), ('signup_modal', 'view', 'signup_modal'), ('signup_weibo', '-unknown-', '-unknown-'), ('signup_weibo_referral', '-unknown-', '-unknown-'), ('similar_listings', 'data', 'similar_listings'), ('similar_listings_v2', '', ''), ('sldf', '-unknown-', '-unknown-'), ('slideshow', '-unknown-', '-unknown-'), ('social', '-unknown-', '-unknown-'), ('social-media', '-unknown-', '-unknown-'), ('social_connections', '-unknown-', '-unknown-'), ('social_connections', 'data', 'user_social_connections'), ('south-america', '-unknown-', '-unknown-'), ('southern-europe', '-unknown-', '-unknown-'), ('special_offer', 'message_post', 'message_post'), ('spoken_languages', 'data', 'user_languages'), ('status', '-unknown-', '-unknown-'), ('stpcv', '-unknown-', '-unknown-'), ('sublets', '-unknown-', '-unknown-'), ('submit_contact', '-unknown-', '-unknown-'), ('support_phone_numbers', '-unknown-', '-unknown-'), ('supported', '-unknown-', '-unknown-'), ('sync', '-unknown-', '-unknown-'), ('tell_a_friend', '-unknown-', '-unknown-'), ('terms', '-unknown-', '-unknown-'), ('terms', 'view', 'terms_and_privacy'), ('terms_and_conditions', '-unknown-', '-unknown-'), ('this_hosting_reviews', 'click', 'listing_reviews_page'), ('this_hosting_reviews_3000', '-unknown-', '-unknown-'), ('toggle_archived_thread', 'click', 'toggle_archived_thread'), ('toggle_availability', '-unknown-', '-unknown-'), ('toggle_starred_thread', 'click', 'toggle_starred_thread'), ('top_destinations', '-unknown-', '-unknown-'), ('tos_2014', 'view', 'tos_2014'), ('tos_confirm', '-unknown-', '-unknown-'), ('track_activity', '', ''), ('track_page_view', '', ''), ('transaction_history','view', 'account_transaction_history'), ('transaction_history_paginated', '-unknown-', '-unknown-'), ('travel', '-unknown-', '-unknown-'), ('travel_plans_current', 'view', 'your_trips'), ('travel_plans_previous', 'view', 'previous_trips'), ('trust', '-unknown-', '-unknown-'), ('unavailabilities', '-unknown-', '-unknown-'), ('unavailabilities', 'data', 'unavailable_dates'), ('united-states', '-unknown-', '-unknown-'), ('unread', '-unknown-', '-unknown-'), ('unsubscribe', '-unknown-', '-unknown-'), ('update', '', ''), ('update', '-unknown-', '-unknown-'), ('update', 'submit', 'update_listing'), ('update', 'submit', 'update_listing_description'), ('update', 'submit', 'update_user'), ('update', 'submit', 'update_user_profile'), ('update_cached', 'data', 'admin_templates'), ('update_country_of_residence', '-unknown-', '-unknown-'), ('update_friends_display', '-unknown-', '-unknown-'), ('update_hide_from_search_engines', '-unknown-', '-unknown-'), ('update_message', '-unknown-', '-unknown-'), ('update_notifications', '-unknown-', '-unknown-'), ('update_reservation_requirements', '-unknown-', '-unknown-'), ('upload', '-unknown-', '-unknown-'), ('uptodate', '', ''), ('use_mobile_site', '-unknown-', '-unknown-'), ('verify', '-unknown-', '-unknown-'), ('view', 'view', 'p3'), ('views', '-unknown-', '-unknown-'), ('views_campaign', '-unknown-','-unknown-'), ('views_campaign_rules', '-unknown-', '-unknown-'), ('webcam_upload', '-unknown-', '-unknown-'), ('weibo_signup_referral_finish', '-unknown-', '-unknown-'), ('why_host', '-unknown-', '-unknown-'), ('widget', '', ''), ('wishlists', '-unknown-', '-unknown-'), ('zendesk_login_jwt', '-unknown-', '-unknown-')]
	distinctactions = ['', '10', '11', '12', '15', 'about_us', 'account', 'active', 'add_guests', 'add_note', 'agree_terms_check', 'agree_terms_uncheck', 'airbnb_picks', 'ajax_check_dates', 'ajax_get_referrals_amt', 'ajax_get_results', 'ajax_google_translate_description', 'ajax_google_translate_reviews', 'ajax_image_upload', 'ajax_ldp', 'ajax_lwlb_contact', 'ajax_payout_edit', 'ajax_payout_options_by_country', 'ajax_photo_widget_form_iframe', 'ajax_price_and_availability', 'ajax_referral_banner_experiment_type', 'ajax_referral_banner_type', 'ajax_refresh_subtotal', 'ajax_statsd', 'apply', 'apply_code', 'apply_coupon_click', 'apply_coupon_error', 'apply_coupon_error_type', 'apply_reservation', 'ask_question', 'at_checkpoint', 'authenticate', 'authorize', 'available', 'badge', 'become_user', 'calendar_tab_inner2', 'callback', 'campaigns', 'cancel', 'cancellation_policies', 'cancellation_policy_click', 'change', 'change_currency', 'click', 'clickthrough', 'collections','complete', 'complete_redirect', 'complete_status', 'concierge', 'confirm_email', 'connect', 'contact_new', 'countries', 'country_options', 'coupon_code_click', 'coupon_field_focus', 'create', 'create_multiple', 'currencies', 'dashboard', 'decision_tree', 'delete', 'department', 'departments', 'destroy', 'detect_fb_session', 'domains', 'edit', 'edit_verification', 'email_itinerary_colorbox', 'email_share', 'email_wishlist', 'endpoint_error', 'facebook_auto_login', 'faq', 'faq_category', 'faq_experiment_ids', 'forgot_password', 'founders', 'friends', 'friends_new', 'glob', 'google_importer', 'guarantee', 'guest_booked_elsewhere', 'handle_vanity_url', 'header_userpic', 'home_safety_terms', 'host_summary', 'hosting_social_proof', 'how_it_works', 'identity', 'image_order', 'impressions', 'index', 'issue', 'itinerary', 'jumio', 'jumio_redirect', 'jumio_token', 'kba', 'kba_update', 'languages_multiselect', 'listing', 'listings', 'localization_settings', 'login', 'login_modal', 'lookup', 'manage_listing', 'message_to_host_change', 'message_to_host_focus', 'mobile_landing_page', 'multi', 'multi_message_attributes', 'my', 'my_listings', 'new', 'new_session', 'notifications', 'open_graph_setting', 'other_hosting_reviews', 'other_hosting_reviews_first', 'overview', 'p4_refund_policy_terms', 'p4_terms', 'patch', 'pay', 'payment_instruments', 'payment_methods', 'payoneer_account_redirect', 'payout_preferences', 'payout_update', 'pending', 'pending_tickets', 'personalize', 'phone_number_widget', 'phone_verification_modal', 'phone_verification_number_submitted_for_call', 'phone_verification_number_submitted_for_sms', 'phone_verification_number_sucessfully_submitted', 'phone_verification_success', 'photography', 'popular', 'popular_listing', 'populate_from_facebook', 'populate_help_dropdown', 'position', 'press_news', 'press_release', 'privacy', 'profile_pic', 'push_notification_callback', 'qt2', 'qt_reply_v2', 'qt_with', 'rate', 'read_policy_click', 'receipt', 'recent_reservations', 'recommend', 'recommendations', 'recommended_listings', 'redirect', 'references', 'referrer_status', 'remove_dashboard_alert', 'reputation', 'request_new_confirm_email', 'requested', 'requirements', 'review_page', 'reviews', 'reviews_new', 'salute', 'search', 'search_results', 'set_password', 'set_user', 'settings', 'show', 'show_code', 'show_personalize', 'signature', 'signed_out_modal', 'signup_login', 'signup_modal', 'similar_listings', 'similar_listings_v2', 'social_connections', 'spoken_languages', 'status', 'submit_contact', 'supported', 'tell_a_friend', 'terms', 'terms_and_conditions', 'this_hosting_reviews', 'toggle_archived_thread', 'toggle_starred_thread', 'top_destinations', 'tos_confirm', 'track_page_view', 'transaction_history', 'transaction_history_paginated', 'travel_plans_current', 'travel_plans_previous', 'trust', 'unavailabilities', 'unread', 'update', 'update_cached', 'update_hide_from_search_engines', 'update_notifications', 'upload', 'uptodate', 'verify', 'webcam_upload', 'why_host']
	dictinctaction_types = np.array(['', '-unknown-', 'booking_request', 'click', 'data', 'message_post', 'partner_callback', 'submit', 'view'])
	distinctaction_details = ['', '-unknown-', 'account_notification_settings', 'account_payout_preferences', 'account_privacy_settings', 'account_transaction_history', 'admin_templates', 'airbnb_picks_wishlists', 'alteration_field', 'alteration_request', 'apply_coupon', 'apply_coupon_click', 'apply_coupon_error', 'at_checkpoint', 'book_it', 'cancellation_policies', 'cancellation_policy_click', 'change_contact_host_dates', 'change_or_alter', 'change_trip_characteristics', 'complete_booking', 'confirm_email', 'confirm_email_link', 'contact_host', 'coupon_code_click', 'coupon_field_focus', 'create_listing', 'create_phone_numbers', 'create_user', 'dashboard', 'delete_listing', 'delete_phone_numbers', 'edit_profile', 'email_wishlist', 'email_wishlist_button', 'forgot_password', 'friends_wishlists', 'guest_cancellation', 'guest_itinerary', 'guest_receipt', 'header_userpic', 'host_guarantee', 'host_home', 'instant_book', 'list_your_space', 'listing_descriptions', 'listing_recommendations', 'listing_reviews', 'listing_reviews_page', 'login', 'login_modal', 'login_page', 'lookup_message_thread', 'manage_listing', 'message_inbox', 'message_post', 'message_thread', 'message_to_host_change', 'message_to_host_focus', 'notifications', 'oauth_login', 'oauth_response', 'p1', 'p3', 'p4', 'p4_refund_policy_terms', 'p4_terms', 'p5', 'payment_instruments', 'pending', 'phone_verification_success', 'popular_wishlists', 'post_checkout_action', 'previous_trips', 'profile_references', 'profile_verifications', 'read_policy_click', 'remove_dashboard_alert', 'request_new_confirm_email', 'request_to_book', 'reservations', 'send_message', 'set_password', 'set_password_page', 'signup', 'signup_login_page', 'signup_modal', 'similar_listings', 'terms_and_privacy', 'toggle_archived_thread', 'toggle_starred_thread', 'translate_listing_reviews', 'translations', 'trip_availability', 'unavailable_dates', 'update_listing', 'update_listing_description','update_user', 'update_user_profile', 'user_friend_recommendations', 'user_languages', 'user_listings', 'user_profile','user_profile_content_update', 'user_reviews', 'user_social_connections', 'user_tax_forms', 'user_wishlists', 'view_listing', 'view_search_results', 'wishlist', 'wishlist_content_update', 'wishlist_note', 'your_listings', 'your_reservations', 'your_trips']
	distinctdevice_types = ['-unknown-', 'Android App Unknown Phone/Tablet', 'Android Phone', 'Blackberry', 'Chromebook', 'Linux Desktop', 'Mac Desktop', 'Tablet', 'Windows Desktop', 'Windows Phone', 'iPad Tablet', 'iPhone', 'iPodtouch']
	
	# row 1 + row 2
	distinct_duo1 = [('', 'message_post'), ('10', 'message_post'), ('11', 'message_post'), ('12', 'message_post'), ('15', 'message_post'), ('about_us', '-unknown-'), ('accept_decline', 'view'), ('account', '-unknown-'), ('acculynk_bin_check_failed', '-unknown-'), ('acculynk_bin_check_success', '-unknown-'), ('acculynk_load_pin_pad', '-unknown-'), ('acculynk_pin_pad_error', '-unknown-'), ('acculynk_pin_pad_inactive', '-unknown-'), ('acculynk_pin_pad_success', '-unknown-'), ('acculynk_session_obtained', '-unknown-'), ('active', '-unknown-'), ('add_business_address_colorbox', '-unknown-'), ('add_guest_colorbox', '-unknown-'), ('add_guests', '-unknown-'), ('add_note', 'submit'), ('agree_terms_check', '-unknown-'), ('agree_terms_uncheck', '-unknown-'), ('airbnb_picks', 'view'), ('airbrb', '-unknown-'), ('ajax_check_dates', 'click'), ('ajax_get_referrals_amt', '-unknown-'), ('ajax_get_results', 'click'), ('ajax_google_translate', '-unknown-'), ('ajax_google_translate_description', '-unknown-'), ('ajax_google_translate_reviews', 'click'), ('ajax_image_upload', '-unknown-'), ('ajax_ldp', '-unknown-'), ('ajax_lwlb_contact', 'click'), ('ajax_payout_edit', '-unknown-'), ('ajax_payout_options_by_country', '-unknown-'), ('ajax_payout_split_edit', '-unknown-'), ('ajax_photo_widget', '-unknown-'), ('ajax_photo_widget_form_iframe', '-unknown-'), ('ajax_price_and_availability', 'click'), ('ajax_referral_banner_experiment_type', '-unknown-'), ('ajax_referral_banner_type', '-unknown-'), ('ajax_refresh_subtotal', 'click'), ('ajax_send_message', '-unknown-'), ('ajax_special_offer_dates_available', 'click'), ('ajax_statsd', '-unknown-'), ('ajax_worth', 'submit'), ('apply', '-unknown-'), ('apply_code', '-unknown-'), ('apply_coupon_click', 'click'), ('apply_coupon_click_success', 'click'), ('apply_coupon_error', 'click'), ('apply_coupon_error_type', '-unknown-'), ('apply_reservation', 'submit'), ('approve','-unknown-'), ('approve', 'submit'), ('ask_question', 'submit'), ('at_checkpoint', 'booking_request'), ('authenticate', 'submit'), ('authenticate', 'view'), ('authorize', '-unknown-'), ('available', '-unknown-'), ('available', 'data'), ('badge', '-unknown-'), ('become_user', '-unknown-'), ('book', 'view'), ('booking', 'booking_response'), ('braintree_client_token', ''), ('business_travel', '-unknown-'), ('calendar_tab_inner2', '-unknown-'), ('callback', 'partner_callback'), ('campaigns', ''), ('campaigns', '-unknown-'), ('cancel', 'submit'), ('cancellation_policies', 'view'), ('cancellation_policy_click', 'click'), ('change', 'view'), ('change_availability', '-unknown-'), ('change_availability', 'submit'), ('change_currency', '-unknown-'), ('change_default_payout', '-unknown-'), ('change_password', 'submit'), ('check', ''), ('city_count', '-unknown-'), ('clear_reservation', '-unknown-'), ('click', '-unknown-'), ('click', 'click'), ('clickthrough', '-unknown-'), ('collections', '-unknown-'), ('collections', 'view'), ('complete', '-unknown-'), ('complete_redirect', '-unknown-'), ('complete_status', '-unknown-'), ('concierge', '-unknown-'), ('confirm_email', '-unknown-'), ('confirm_email', 'click'), ('confirmation', '-unknown-'), ('connect', '-unknown-'), ('connect', 'submit'), ('contact_new', '-unknown-'), ('countries', '-unknown-'), ('country_options', '-unknown-'), ('coupon_code_click', 'click'), ('coupon_field_focus', 'click'), ('create', '-unknown-'), ('create', 'submit'), ('create', 'view'), ('create_ach', '-unknown-'), ('create_airbnb', '-unknown-'), ('create_multiple', '-unknown-'), ('create_paypal', '-unknown-'), ('currencies', ''), ('currencies', '-unknown-'), ('custom_recommended_destinations', '-unknown-'), ('dashboard', 'view'), ('deactivate', '-unknown-'), ('deactivated', 'view'), ('deauthorize', '-unknown-'), ('decision_tree', '-unknown-'), ('delete', '-unknown-'), ('delete', 'submit'), ('department', '-unknown-'), ('departments', '-unknown-'), ('desks', '-unknown-'), ('destroy', '-unknown-'), ('destroy', 'submit'), ('detect_fb_session', '-unknown-'), ('disaster_action', ''), ('domains', '-unknown-'), ('edit', '-unknown-'), ('edit', 'view'), ('edit_verification', 'view'), ('email_by_key', '-unknown-'), ('email_itinerary_colorbox', '-unknown-'), ('email_share', 'submit'), ('email_wishlist', 'click'), ('endpoint_error', '-unknown-'), ('envoy_bank_details_redirect', '-unknown-'), ('envoy_form', '-unknown-'), ('events', '-unknown-'), ('facebook_auto_login', '-unknown-'), ('faq', '-unknown-'), ('faq_category', '-unknown-'), ('faq_experiment_ids', '-unknown-'), ('feed', '-unknown-'), ('forgot_password', 'click'), ('forgot_password', 'submit'), ('founders', '-unknown-'), ('friend_listing', '-unknown-'), ('friends', 'view'), ('friends_new', '-unknown-'), ('glob', '-unknown-'), ('google_importer', '-unknown-'), ('guarantee', 'view'), ('guest_billing_receipt', '-unknown-'), ('guest_booked_elsewhere', 'message_post'), ('handle_vanity_url', '-unknown-'), ('hard_fallback_submit', '-unknown-'), ('has_profile_pic', '-unknown-'), ('header_userpic', 'data'), ('home_safety_landing', '-unknown-'), ('home_safety_terms', '-unknown-'), ('hospitality', '-unknown-'), ('hospitality_standards', '-unknown-'), ('host_2013', '-unknown-'), ('host_cancel', '-unknown-'), ('host_summary', '-unknown-'), ('host_summary', 'view'), ('hosting_social_proof', '-unknown-'), ('how_it_works', '-unknown-'), ('identity', '-unknown-'), ('image_order', '-unknown-'), ('impressions', 'view'), ('index', ''), ('index', '-unknown-'), ('index', 'data'), ('index', 'view'), ('invalid_action', '-unknown-'), ('issue', '-unknown-'), ('itinerary', 'view'), ('jumio', '-unknown-'), ('jumio_redirect', '-unknown-'), ('jumio_token', '-unknown-'), ('kba', '-unknown-'), ('kba_update', '-unknown-'), ('languages_multiselect', '-unknown-'), ('life', '-unknown-'), ('listing', 'view'), ('listings', '-unknown-'), ('listings', 'view'), ('load_more', '-unknown-'), ('locale_from_host', '-unknown-'), ('localization_settings', ''), ('localization_settings', '-unknown-'), ('localized', '-unknown-'), ('locations', '-unknown-'), ('login', '-unknown-'), ('login', 'view'), ('login_modal', 'view'), ('lookup', ''), ('manage_listing', 'view'), ('maybe_information', 'message_post'), ('media_resources', '-unknown-'), ('message','-unknown-'), ('message_to_host_change', 'click'), ('message_to_host_focus', 'click'), ('mobile_landing_page', '-unknown-'), ('mobile_oauth_callback', '-unknown-'), ('multi', '-unknown-'), ('multi_message', 'message_post'), ('multi_message_attributes', '-unknown-'), ('my', 'view'), ('my_listings', 'view'), ('my_reservations', 'view'), ('new', '-unknown-'), ('new', 'view'), ('new_host', '-unknown-'), ('new_session', '-unknown-'), ('notifications', '-unknown-'), ('notifications', 'data'), ('notifications', 'submit'), ('notifications', 'view'), ('nyan', '-unknown-'), ('office_location', '-unknown-'), ('onenight', '-unknown-'), ('open_graph_setting', '-unknown-'), ('open_hard_fallback_modal', '-unknown-'), ('other_hosting_reviews', '-unknown-'), ('other_hosting_reviews_first', '-unknown-'), ('overview', '-unknown-'), ('p4_refund_policy_terms', 'click'), ('p4_terms', 'click'), ('patch', '-unknown-'), ('patch', 'modify'), ('pay', '-unknown-'), ('payment_instruments', '-unknown-'), ('payment_instruments', 'data'), ('payment_methods', '-unknown-'), ('payoneer_account_redirect', '-unknown-'), ('payoneer_signup_complete', '-unknown-'), ('payout_delete', '-unknown-'), ('payout_preferences', 'view'), ('payout_update', '-unknown-'), ('pending', '-unknown-'), ('pending', 'booking_request'), ('pending_tickets', '-unknown-'), ('personalize', 'data'), ('phone_number_widget', '-unknown-'), ('phone_verification', ''), ('phone_verification_call_taking_too_long', '-unknown-'), ('phone_verification_error', '-unknown-'), ('phone_verification_modal', '-unknown-'), ('phone_verification_number_submitted_for_call', '-unknown-'), ('phone_verification_number_submitted_for_sms', '-unknown-'), ('phone_verification_number_sucessfully_submitted', '-unknown-'), ('phone_verification_phone_number_removed', '-unknown-'), ('phone_verification_success', 'click'), ('photography', '-unknown-'), ('photography_update', '-unknown-'), ('place_worth', 'view'), ('plaxo_cb', '-unknown-'), ('popular', 'view'), ('popular_listing', '-unknown-'), ('populate_from_facebook', '-unknown-'), ('populate_help_dropdown', '-unknown-'), ('position', '-unknown-'), ('preapproval', 'message_post'), ('press_content', '-unknown-'), ('press_news', '-unknown-'), ('press_release', '-unknown-'), ('pricing', '-unknown-'), ('print_confirmation', '-unknown-'), ('privacy', 'view'), ('profile_pic', '-unknown-'), ('push_notification_callback', '-unknown-'), ('qt2', 'view'), ('qt_reply_v2', '-unknown-'), ('qt_reply_v2', 'submit'), ('qt_with', 'data'), ('questions', '-unknown-'), ('rate', '-unknown-'), ('reactivate', '-unknown-'), ('read_policy_click', 'click'), ('receipt', 'view'), ('recent_reservations', '-unknown-'), ('recommend', '-unknown-'), ('recommendation_page', '-unknown-'), ('recommendations', '-unknown-'), ('recommendations', 'data'), ('recommended_listings', '-unknown-'), ('redirect', '-unknown-'), ('references', 'view'), ('referrer_status', '-unknown-'), ('refund_guest_cancellation', 'submit'), ('relationship', '-unknown-'), ('remove_dashboard_alert', '-unknown-'), ('remove_dashboard_alert', 'click'), ('rentals', '-unknown-'), ('report', '-unknown-'), ('reputation', '-unknown-'), ('request_new_confirm_email', 'click'), ('request_photography', '-unknown-'),('requested', 'submit'), ('requested', 'view'), ('requirements', '-unknown-'), ('reservation', '-unknown-'), ('reset_calendar', '-unknown-'), ('respond', 'submit'), ('rest-of-world', '-unknown-'), ('revert_to_admin', '-unknown-'), ('review_page', '-unknown-'), ('reviews', '-unknown-'), ('reviews', 'data'), ('reviews', 'view'), ('reviews_new', '-unknown-'), ('salute', '-unknown-'), ('sandy', '-unknown-'), ('satisfy', ''), ('search', '-unknown-'), ('search', 'click'), ('search_results', 'click'), ('set_default', '-unknown-'), ('set_default', 'submit'), ('set_minimum_payout_amount', '-unknown-'), ('set_password', 'submit'), ('set_password', 'view'), ('set_user', 'submit'), ('settings', '-unknown-'), ('show', ''), ('show', '-unknown-'), ('show', 'data'), ('show', 'view'), ('show_code', '-unknown-'), ('show_personalize', 'data'), ('signature', '-unknown-'), ('signed_out_modal', ''), ('signup_login', 'view'), ('signup_modal', 'view'), ('signup_weibo', '-unknown-'), ('signup_weibo_referral', '-unknown-'), ('similar_listings', 'data'), ('similar_listings_v2', ''), ('sldf', '-unknown-'), ('slideshow', '-unknown-'), ('social', '-unknown-'), ('social-media', '-unknown-'), ('social_connections', '-unknown-'), ('social_connections', 'data'), ('south-america', '-unknown-'), ('southern-europe', '-unknown-'), ('special_offer', 'message_post'), ('spoken_languages', 'data'), ('status', '-unknown-'), ('stpcv', '-unknown-'), ('sublets', '-unknown-'), ('submit_contact', '-unknown-'), ('support_phone_numbers', '-unknown-'), ('supported', '-unknown-'), ('sync', '-unknown-'), ('tell_a_friend', '-unknown-'), ('terms', '-unknown-'), ('terms', 'view'), ('terms_and_conditions', '-unknown-'), ('this_hosting_reviews', 'click'), ('this_hosting_reviews_3000', '-unknown-'), ('toggle_archived_thread', 'click'), ('toggle_availability', '-unknown-'), ('toggle_starred_thread', 'click'), ('top_destinations', '-unknown-'), ('tos_2014', 'view'), ('tos_confirm', '-unknown-'), ('track_activity', ''), ('track_page_view', ''), ('transaction_history', 'view'), ('transaction_history_paginated', '-unknown-'), ('travel', '-unknown-'), ('travel_plans_current', 'view'), ('travel_plans_previous', 'view'), ('trust', '-unknown-'), ('unavailabilities', '-unknown-'), ('unavailabilities', 'data'), ('united-states', '-unknown-'), ('unread', '-unknown-'), ('unsubscribe', '-unknown-'), ('update', ''), ('update', '-unknown-'), ('update', 'submit'), ('update_cached', 'data'), ('update_country_of_residence', '-unknown-'), ('update_friends_display', '-unknown-'), ('update_hide_from_search_engines', '-unknown-'), ('update_message', '-unknown-'), ('update_notifications', '-unknown-'), ('update_reservation_requirements', '-unknown-'), ('upload', '-unknown-'), ('uptodate', ''), ('use_mobile_site', '-unknown-'), ('verify', '-unknown-'), ('view', 'view'), ('views', '-unknown-'), ('views_campaign', '-unknown-'), ('views_campaign_rules', '-unknown-'), ('webcam_upload', '-unknown-'), ('weibo_signup_referral_finish', '-unknown-'), ('why_host', '-unknown-'), ('widget', ''), ('wishlists', '-unknown-'), ('zendesk_login_jwt', '-unknown-')]
	# row 1 + row 3
	distinct_duo2 = [('', 'message_post'), ('10', 'message_post'), ('11', 'message_post'), ('12', 'message_post'), ('15', 'message_post'), ('about_us', '-unknown-'), ('accept_decline', 'host_respond_page'), ('account', '-unknown-'), ('acculynk_bin_check_failed', '-unknown-'), ('acculynk_bin_check_success', '-unknown-'), ('acculynk_load_pin_pad', '-unknown-'), ('acculynk_pin_pad_error', '-unknown-'), ('acculynk_pin_pad_inactive', '-unknown-'), ('acculynk_pin_pad_success', '-unknown-'), ('acculynk_session_obtained', '-unknown-'), ('active', '-unknown-'), ('add_business_address_colorbox', '-unknown-'), ('add_guest_colorbox', '-unknown-'), ('add_guests', '-unknown-'), ('add_note', 'wishlist_note'), ('agree_terms_check', '-unknown-'), ('agree_terms_uncheck', '-unknown-'), ('airbnb_picks', 'airbnb_picks_wishlists'), ('airbrb','-unknown-'), ('ajax_check_dates', 'change_contact_host_dates'), ('ajax_get_referrals_amt', '-unknown-'), ('ajax_get_results', 'view_search_results'), ('ajax_google_translate', '-unknown-'), ('ajax_google_translate_description', '-unknown-'), ('ajax_google_translate_reviews', 'translate_listing_reviews'), ('ajax_image_upload', '-unknown-'), ('ajax_ldp', '-unknown-'), ('ajax_lwlb_contact', 'contact_host'), ('ajax_payout_edit', '-unknown-'), ('ajax_payout_options_by_country', '-unknown-'), ('ajax_payout_split_edit', '-unknown-'), ('ajax_photo_widget', '-unknown-'), ('ajax_photo_widget_form_iframe', '-unknown-'), ('ajax_price_and_availability', 'alteration_field'), ('ajax_referral_banner_experiment_type', '-unknown-'), ('ajax_referral_banner_type', '-unknown-'), ('ajax_refresh_subtotal', 'change_trip_characteristics'), ('ajax_send_message', '-unknown-'), ('ajax_special_offer_dates_available', 'special_offer_field'), ('ajax_statsd', '-unknown-'), ('ajax_worth', 'calculate_worth'), ('apply', '-unknown-'), ('apply_code', '-unknown-'), ('apply_coupon_click', 'apply_coupon_click'), ('apply_coupon_click_success', 'apply_coupon_click_success'), ('apply_coupon_error', 'apply_coupon_error'), ('apply_coupon_error_type', '-unknown-'), ('apply_reservation', 'apply_coupon'), ('approve', '-unknown-'), ('approve', 'host_respond'), ('ask_question', 'contact_host'), ('at_checkpoint', 'at_checkpoint'), ('authenticate', 'login'), ('authenticate', 'login_page'), ('authorize', '-unknown-'), ('available', '-unknown-'), ('available', 'trip_availability'), ('badge', '-unknown-'), ('become_user', '-unknown-'), ('book', 'p4'),('booking', 'booking'), ('braintree_client_token', ''), ('business_travel', '-unknown-'), ('calendar_tab_inner2', '-unknown-'), ('callback', 'oauth_response'), ('campaigns', ''), ('campaigns', '-unknown-'), ('cancel', 'guest_cancellation'), ('cancellation_policies', 'cancellation_policies'), ('cancellation_policy_click', 'cancellation_policy_click'), ('change', 'change_or_alter'), ('change_availability', '-unknown-'), ('change_availability', 'change_availability'), ('change_currency', '-unknown-'), ('change_default_payout', '-unknown-'), ('change_password', 'change_password'), ('check', ''), ('city_count', '-unknown-'), ('clear_reservation', '-unknown-'), ('click', '-unknown-'), ('click', 'book_it'), ('click', 'cancellation_policy'), ('click', 'click_about_host'), ('click', 'click_amenities'), ('click', 'click_reviews'), ('click', 'complete_booking'), ('click', 'contact_host'), ('click', 'instant_book'), ('click', 'move_map'), ('click', 'photos'), ('click', 'request_to_book'), ('click', 'share'), ('clickthrough', '-unknown-'), ('collections', '-unknown-'), ('collections', 'user_wishlists'), ('complete', '-unknown-'), ('complete_redirect', '-unknown-'), ('complete_status', '-unknown-'), ('concierge', '-unknown-'), ('confirm_email', '-unknown-'), ('confirm_email', 'confirm_email'), ('confirm_email', 'confirm_email_link'), ('confirmation', '-unknown-'), ('connect', '-unknown-'), ('connect', 'oauth_login'), ('contact_new', '-unknown-'), ('countries', '-unknown-'), ('country_options', '-unknown-'), ('coupon_code_click', 'coupon_code_click'), ('coupon_field_focus', 'coupon_field_focus'), ('create', '-unknown-'), ('create', 'create_alteration_request'), ('create', 'create_listing'), ('create', 'create_payment_instrument'), ('create', 'create_phone_numbers'), ('create', 'create_user'), ('create', 'list_your_space'), ('create', 'signup'), ('create_ach', '-unknown-'), ('create_airbnb', '-unknown-'), ('create_multiple', '-unknown-'), ('create_paypal', '-unknown-'), ('currencies', ''), ('currencies', '-unknown-'), ('custom_recommended_destinations', '-unknown-'), ('dashboard', 'dashboard'), ('deactivate', '-unknown-'), ('deactivated', 'host_standard_suspension'), ('deauthorize', '-unknown-'), ('decision_tree', '-unknown-'), ('delete', '-unknown-'), ('delete', 'deactivate_user_account'), ('delete', 'delete_listing'), ('delete', 'delete_listing_description'), ('delete', 'delete_phone_numbers'), ('delete', 'phone_numbers'), ('delete', 'reservations'), ('department', '-unknown-'), ('departments', '-unknown-'), ('desks', '-unknown-'), ('destroy', '-unknown-'), ('destroy', 'delete_payment_instrument'), ('detect_fb_session', '-unknown-'), ('disaster_action', ''), ('domains', '-unknown-'), ('edit', '-unknown-'), ('edit', 'edit_profile'), ('edit_verification', 'profile_verifications'), ('email_by_key', '-unknown-'), ('email_itinerary_colorbox', '-unknown-'), ('email_share', 'email_wishlist'), ('email_wishlist', 'email_wishlist_button'), ('endpoint_error', '-unknown-'), ('envoy_bank_details_redirect', '-unknown-'), ('envoy_form', '-unknown-'), ('events', '-unknown-'), ('facebook_auto_login', '-unknown-'), ('faq', '-unknown-'), ('faq_category', '-unknown-'), ('faq_experiment_ids', '-unknown-'), ('feed', '-unknown-'), ('forgot_password', 'forgot_password'), ('founders', '-unknown-'), ('friend_listing', '-unknown-'), ('friends', 'friends_wishlists'), ('friends_new', '-unknown-'), ('glob', '-unknown-'), ('google_importer', '-unknown-'), ('guarantee', 'host_guarantee'), ('guest_billing_receipt', '-unknown-'), ('guest_booked_elsewhere', 'message_post'), ('handle_vanity_url', '-unknown-'), ('hard_fallback_submit', '-unknown-'), ('has_profile_pic', '-unknown-'), ('header_userpic', 'header_userpic'), ('home_safety_landing', '-unknown-'), ('home_safety_terms', '-unknown-'), ('hospitality', '-unknown-'), ('hospitality_standards', '-unknown-'), ('host_2013', '-unknown-'), ('host_cancel', '-unknown-'), ('host_summary', '-unknown-'), ('host_summary', 'host_home'), ('hosting_social_proof', '-unknown-'), ('how_it_works', '-unknown-'), ('identity', '-unknown-'), ('image_order', '-unknown-'), ('impressions', 'p4'), ('index', ''), ('index', '-unknown-'), ('index', 'account_payment_methods'), ('index', 'homepage'), ('index', 'listing_descriptions'), ('index', 'message_inbox'), ('index', 'message_thread'), ('index', 'reservations'), ('index', 'user_tax_forms'), ('index', 'user_wishlists'), ('index', 'view_ghosting_reasons'), ('index', 'view_ghostings'), ('index', 'view_identity_verifications'), ('index', 'view_locations'), ('index', 'view_reservations'), ('index', 'view_resolutions'), ('index', 'view_search_results'), ('index', 'view_user_real_names'), ('index', 'your_listings'), ('invalid_action', '-unknown-'), ('issue', '-unknown-'), ('itinerary', 'guest_itinerary'), ('jumio', '-unknown-'), ('jumio_redirect', '-unknown-'), ('jumio_token', '-unknown-'), ('kba', '-unknown-'), ('kba_update', '-unknown-'), ('languages_multiselect', '-unknown-'), ('life', '-unknown-'), ('listing', 'p3'), ('listings', '-unknown-'), ('listings', 'user_listings'), ('load_more', '-unknown-'), ('locale_from_host', '-unknown-'), ('localization_settings', ''), ('localization_settings', '-unknown-'), ('localized', '-unknown-'), ('locations', '-unknown-'), ('login', '-unknown-'), ('login', 'login_page'), ('login_modal', 'login_modal'), ('lookup', ''), ('manage_listing', 'manage_listing'),('maybe_information', 'message_post'), ('media_resources', '-unknown-'), ('message', '-unknown-'), ('message_to_host_change', 'message_to_host_change'), ('message_to_host_focus', 'message_to_host_focus'), ('mobile_landing_page', '-unknown-'), ('mobile_oauth_callback', '-unknown-'), ('multi', '-unknown-'), ('multi_message', 'message_post'), ('multi_message_attributes', '-unknown-'), ('my', 'user_wishlists'), ('my_listings', 'your_reservations'), ('my_reservations', 'your_reservations'), ('new', '-unknown-'), ('new', 'list_your_space'), ('new_host', '-unknown-'), ('new_session', '-unknown-'), ('notifications', '-unknown-'), ('notifications', 'account_notification_settings'), ('notifications', 'notifications'), ('nyan', '-unknown-'), ('office_location', '-unknown-'), ('onenight', '-unknown-'), ('open_graph_setting', '-unknown-'), ('open_hard_fallback_modal', '-unknown-'), ('other_hosting_reviews', '-unknown-'), ('other_hosting_reviews_first', '-unknown-'), ('overview', '-unknown-'), ('p4_refund_policy_terms', 'p4_refund_policy_terms'), ('p4_terms', 'p4_terms'), ('patch', '-unknown-'), ('patch', 'modify_reservations'), ('patch', 'modify_users'), ('pay', '-unknown-'), ('payment_instruments', '-unknown-'), ('payment_instruments', 'payment_instruments'), ('payment_methods', '-unknown-'), ('payoneer_account_redirect', '-unknown-'), ('payoneer_signup_complete', '-unknown-'), ('payout_delete', '-unknown-'), ('payout_preferences', 'account_payout_preferences'), ('payout_update', '-unknown-'), ('pending', '-unknown-'), ('pending', 'pending'), ('pending_tickets', '-unknown-'), ('personalize', 'wishlist_content_update'), ('phone_number_widget', '-unknown-'), ('phone_verification', ''), ('phone_verification_call_taking_too_long', '-unknown-'), ('phone_verification_error', '-unknown-'), ('phone_verification_modal', '-unknown-'), ('phone_verification_number_submitted_for_call', '-unknown-'), ('phone_verification_number_submitted_for_sms', '-unknown-'), ('phone_verification_number_sucessfully_submitted', '-unknown-'), ('phone_verification_phone_number_removed', '-unknown-'), ('phone_verification_success', 'phone_verification_success'), ('photography', '-unknown-'), ('photography_update', '-unknown-'), ('place_worth', 'place_worth'), ('plaxo_cb', '-unknown-'), ('popular', 'popular_wishlists'), ('popular_listing', '-unknown-'), ('populate_from_facebook', '-unknown-'), ('populate_help_dropdown', '-unknown-'), ('position', '-unknown-'), ('preapproval', 'message_post'), ('press_content', '-unknown-'), ('press_news', '-unknown-'), ('press_release', '-unknown-'), ('pricing', '-unknown-'), ('print_confirmation', '-unknown-'), ('privacy', 'account_privacy_settings'), ('profile_pic', '-unknown-'), ('push_notification_callback', '-unknown-'), ('qt2', 'message_thread'), ('qt_reply_v2', '-unknown-'), ('qt_reply_v2', 'send_message'), ('qt_with', 'lookup_message_thread'), ('questions', '-unknown-'), ('rate', '-unknown-'), ('reactivate', '-unknown-'), ('read_policy_click', 'read_policy_click'), ('receipt', 'guest_receipt'), ('recent_reservations', '-unknown-'), ('recommend', '-unknown-'), ('recommendation_page', '-unknown-'), ('recommendations', '-unknown-'), ('recommendations', 'listing_recommendations'), ('recommendations','user_friend_recommendations'), ('recommended_listings', '-unknown-'), ('redirect', '-unknown-'), ('references', 'profile_references'), ('referrer_status', '-unknown-'), ('refund_guest_cancellation', 'host_refund_guest'), ('relationship', '-unknown-'), ('remove_dashboard_alert', '-unknown-'), ('remove_dashboard_alert', 'remove_dashboard_alert'), ('rentals', '-unknown-'), ('report', '-unknown-'), ('reputation', '-unknown-'), ('request_new_confirm_email', 'request_new_confirm_email'), ('request_photography', '-unknown-'), ('requested', 'p5'), ('requested', 'post_checkout_action'), ('requirements', '-unknown-'), ('reservation', '-unknown-'), ('reset_calendar', '-unknown-'), ('respond', 'respond_to_alteration_request'), ('rest-of-world', '-unknown-'), ('revert_to_admin', '-unknown-'), ('review_page', '-unknown-'), ('reviews', '-unknown-'), ('reviews', 'listing_reviews'), ('reviews', 'profile_reviews'), ('reviews', 'user_reviews'), ('reviews_new', '-unknown-'), ('salute', '-unknown-'), ('sandy', '-unknown-'), ('satisfy', ''), ('search', '-unknown-'), ('search', 'view_search_results'), ('search_results', 'view_search_results'), ('set_default', '-unknown-'), ('set_default', 'set_default_payment_instrument'), ('set_minimum_payout_amount', '-unknown-'), ('set_password', 'set_password'), ('set_password', 'set_password_page'), ('set_user', 'create_listing'), ('settings', '-unknown-'), ('show', ''), ('show', '-unknown-'), ('show', 'alteration_request'), ('show', 'p1'), ('show', 'p3'), ('show', 'translations'), ('show', 'user_profile'), ('show', 'view_identity_verifications'), ('show', 'view_listing'), ('show','view_security_checks'), ('show', 'wishlist'), ('show_code', '-unknown-'), ('show_personalize', 'user_profile_content_update'), ('signature', '-unknown-'), ('signed_out_modal', ''), ('signup_login', 'signup_login_page'), ('signup_modal', 'signup_modal'), ('signup_weibo', '-unknown-'), ('signup_weibo_referral', '-unknown-'), ('similar_listings', 'similar_listings'), ('similar_listings_v2', ''), ('sldf', '-unknown-'), ('slideshow', '-unknown-'), ('social', '-unknown-'), ('social-media', '-unknown-'), ('social_connections', '-unknown-'), ('social_connections', 'user_social_connections'), ('south-america', '-unknown-'), ('southern-europe', '-unknown-'), ('special_offer', 'message_post'), ('spoken_languages', 'user_languages'), ('status', '-unknown-'), ('stpcv', '-unknown-'), ('sublets', '-unknown-'), ('submit_contact', '-unknown-'), ('support_phone_numbers', '-unknown-'), ('supported', '-unknown-'), ('sync', '-unknown-'), ('tell_a_friend', '-unknown-'), ('terms', '-unknown-'), ('terms', 'terms_and_privacy'), ('terms_and_conditions', '-unknown-'), ('this_hosting_reviews','listing_reviews_page'), ('this_hosting_reviews_3000', '-unknown-'), ('toggle_archived_thread', 'toggle_archived_thread'), ('toggle_availability', '-unknown-'), ('toggle_starred_thread', 'toggle_starred_thread'), ('top_destinations', '-unknown-'), ('tos_2014', 'tos_2014'), ('tos_confirm', '-unknown-'), ('track_activity', ''), ('track_page_view', ''), ('transaction_history', 'account_transaction_history'), ('transaction_history_paginated', '-unknown-'), ('travel', '-unknown-'), ('travel_plans_current', 'your_trips'), ('travel_plans_previous', 'previous_trips'), ('trust', '-unknown-'), ('unavailabilities', '-unknown-'), ('unavailabilities', 'unavailable_dates'), ('united-states', '-unknown-'), ('unread', '-unknown-'), ('unsubscribe', '-unknown-'), ('update', ''), ('update', '-unknown-'), ('update', 'update_listing'), ('update', 'update_listing_description'), ('update', 'update_user'), ('update', 'update_user_profile'), ('update_cached', 'admin_templates'), ('update_country_of_residence', '-unknown-'), ('update_friends_display', '-unknown-'), ('update_hide_from_search_engines', '-unknown-'), ('update_message', '-unknown-'), ('update_notifications', '-unknown-'), ('update_reservation_requirements', '-unknown-'), ('upload', '-unknown-'), ('uptodate', ''), ('use_mobile_site', '-unknown-'), ('verify', '-unknown-'), ('view', 'p3'), ('views', '-unknown-'), ('views_campaign', '-unknown-'), ('views_campaign_rules', '-unknown-'), ('webcam_upload', '-unknown-'), ('weibo_signup_referral_finish', '-unknown-'), ('why_host', '-unknown-'), ('widget', ''), ('wishlists', '-unknown-'), ('zendesk_login_jwt', '-unknown-')]
	# row 2 + row 3
	distinct_duo3 = [('', ''), ('-unknown-', '-unknown-'), ('-unknown-', 'phone_numbers'), ('-unknown-', 'reservations'), ('booking_request', 'at_checkpoint'), ('booking_request', 'pending'), ('booking_response', 'booking'), ('click', 'alteration_field'), ('click', 'apply_coupon_click'), ('click', 'apply_coupon_click_success'), ('click', 'apply_coupon_error'), ('click', 'book_it'), ('click', 'cancellation_policy'), ('click', 'cancellation_policy_click'), ('click', 'change_contact_host_dates'), ('click', 'change_trip_characteristics'), ('click', 'click_about_host'), ('click', 'click_amenities'), ('click', 'click_reviews'), ('click', 'complete_booking'), ('click', 'confirm_email'), ('click', 'confirm_email_link'), ('click', 'contact_host'), ('click', 'coupon_code_click'), ('click', 'coupon_field_focus'), ('click', 'email_wishlist_button'), ('click', 'forgot_password'), ('click', 'instant_book'), ('click', 'listing_reviews_page'), ('click', 'message_to_host_change'), ('click', 'message_to_host_focus'), ('click', 'move_map'), ('click', 'p4_refund_policy_terms'), ('click', 'p4_terms'), ('click', 'phone_verification_success'), ('click', 'photos'), ('click', 'read_policy_click'), ('click', 'remove_dashboard_alert'), ('click', 'request_new_confirm_email'), ('click', 'request_to_book'), ('click', 'share'), ('click', 'special_offer_field'), ('click', 'toggle_archived_thread'), ('click', 'toggle_starred_thread'), ('click', 'translate_listing_reviews'), ('click', 'view_search_results'), ('data', 'admin_templates'), ('data', 'header_userpic'), ('data', 'listing_recommendations'), ('data', 'listing_reviews'), ('data', 'lookup_message_thread'), ('data', 'notifications'), ('data', 'payment_instruments'), ('data', 'reservations'), ('data', 'similar_listings'), ('data', 'translations'), ('data', 'trip_availability'), ('data', 'unavailable_dates'), ('data', 'user_friend_recommendations'), ('data', 'user_languages'), ('data', 'user_profile_content_update'), ('data', 'user_reviews'), ('data', 'user_social_connections'), ('data', 'user_tax_forms'), ('data', 'wishlist_content_update'), ('message_post', 'message_post'), ('modify', 'modify_reservations'), ('modify', 'modify_users'), ('partner_callback', 'oauth_response'), ('submit', 'apply_coupon'), ('submit', 'calculate_worth'), ('submit', 'change_availability'), ('submit', 'change_password'), ('submit', 'contact_host'), ('submit', 'create_alteration_request'), ('submit', 'create_listing'), ('submit', 'create_payment_instrument'), ('submit', 'create_phone_numbers'), ('submit', 'create_user'), ('submit', 'deactivate_user_account'), ('submit', 'delete_listing'), ('submit', 'delete_listing_description'), ('submit', 'delete_payment_instrument'), ('submit', 'delete_phone_numbers'), ('submit', 'email_wishlist'), ('submit', 'forgot_password'), ('submit', 'guest_cancellation'), ('submit', 'host_refund_guest'), ('submit', 'host_respond'), ('submit', 'login'), ('submit', 'notifications'), ('submit', 'oauth_login'), ('submit', 'post_checkout_action'), ('submit', 'respond_to_alteration_request'), ('submit', 'send_message'), ('submit', 'set_default_payment_instrument'), ('submit', 'set_password'), ('submit', 'signup'), ('submit', 'update_listing'), ('submit', 'update_listing_description'), ('submit', 'update_user'), ('submit', 'update_user_profile'), ('submit', 'wishlist_note'), ('view', 'account_notification_settings'), ('view', 'account_payment_methods'), ('view', 'account_payout_preferences'), ('view', 'account_privacy_settings'), ('view', 'account_transaction_history'), ('view', 'airbnb_picks_wishlists'), ('view', 'alteration_request'), ('view', 'cancellation_policies'), ('view', 'change_or_alter'), ('view', 'dashboard'), ('view', 'edit_profile'), ('view', 'friends_wishlists'), ('view', 'guest_itinerary'), ('view', 'guest_receipt'), ('view', 'homepage'), ('view', 'host_guarantee'), ('view', 'host_home'), ('view', 'host_respond_page'), ('view', 'host_standard_suspension'), ('view', 'list_your_space'), ('view', 'listing_descriptions'), ('view', 'login_modal'), ('view', 'login_page'), ('view', 'manage_listing'), ('view', 'message_inbox'), ('view', 'message_thread'), ('view', 'p1'), ('view', 'p3'), ('view', 'p4'), ('view', 'p5'), ('view', 'place_worth'), ('view', 'popular_wishlists'), ('view', 'previous_trips'), ('view', 'profile_references'), ('view', 'profile_reviews'), ('view', 'profile_verifications'), ('view', 'set_password_page'), ('view', 'signup_login_page'), ('view', 'signup_modal'), ('view', 'terms_and_privacy'), ('view', 'tos_2014'), ('view', 'user_listings'), ('view', 'user_profile'), ('view', 'user_wishlists'), ('view', 'view_ghosting_reasons'), ('view', 'view_ghostings'), ('view', 'view_identity_verifications'), ('view', 'view_listing'), ('view', 'view_locations'), ('view', 'view_reservations'), ('view', 'view_resolutions'), ('view', 'view_search_results'), ('view', 'view_security_checks'), ('view', 'view_user_real_names'), ('view', 'wishlist'), ('view', 'your_listings'), ('view', 'your_reservations'), ('view', 'your_trips')]
	
	# User has session?
	hassession = np.array([1 if id in keyswithsessions else -1 for id in ids])
	numusers[:,starti] = hassession
	starti += 1
	
	# User has booking request?
	#hasbr = np.array([1 if id in keyswithbrs else -1 for id in ids])
	#numusers[:,starti] = hasbr
	#starti += 1
	
	print max(len(x) for x in sessions.values())
	
	"""
	sessiontabs = dict()
	# One-hot for every session
	idi = 0
	for id in ids:
		if id in sessions:
			idsess = sessions[id]
			tab = np.load('sessiontabs/' + id + '.npy')
			
			"#""
			tab = np.empty((len(idsess), 9))
			sessi = 0
			for row in idsess:
				# Booking request type
				tab[sessi,0:9] = scaleCat(dictinctaction_types == row[2])
				sessi += 1
			sessiontabs[id] = tab
			
			np.save('sessiontabs/' + id + '.npy', tab)
			"#""
			sessiontabs[id] = tab
			
			if idi % 1000 == 0:
				print idi,"/",len(ids)
		idi += 1
	import cPickle
	with open('sessiontabs.pickle', 'wb') as f:
		cPickle.dump(sessiontabs, f)
	"""
	i = 0
	for id in ids:
		if id in sessions:
			# sessions for this id
			sess = sessions[id]
			# session triads
			sesstriads = set()
			sessactions = set()
			sesstypes = set()
			sessdetails = set()
			sessduo1s = set()
			sessduo2s = set()
			sessduo3s = set()
			for row in sess:
				sesstriads.add((row[1], row[2], row[3]))
				sessactions.add(row[1])
				sesstypes.add(row[2])
				sessdetails.add(row[3])
				sessduo1s.add((row[1], row[2]))
				sessduo2s.add((row[1], row[3]))
				sessduo3s.add((row[2], row[3]))
			#sesstriads = dict()
			#for row in sess:
			#	if (row[1], row[2], row[3]) not in sesstriads:
			#		sesstriads[(row[1], row[2], row[3])] = 0
			#	if row[5] != '':
			#		sesstriads[(row[1], row[2], row[3])] += float(row[5])
		else:
			sesstriads = set()
			sessactions = set()
			sesstypes = set()
			sessdetails = set()
		# one hot encode
		j = starti
		for triad in distincttriads:
			if triad in sesstriads:
				#numusers[i,j] = scaledWmm(sesstriads[triad], 0, 100000.0)
				numusers[i,j] = 1
				#print numusers[i,j]
			else:
				numusers[i,j] = -1
				#numusers[i,j+1] = -1
			j += 1
		for action in distinctactions:
			if action in sessactions:
				numusers[i,j] = 1
			else:
				numusers[i,j] = -1
			j += 1
		for type in dictinctaction_types:
			if type in sesstypes:
				numusers[i,j] = 1
			else:
				numusers[i,j] = -1
			j += 1
		for detail in distinctaction_details:
			if detail in sessdetails:
				numusers[i,j] = 1
			else:
				numusers[i,j] = -1
			j += 1
		# TODO: add device type
		i += 1
	starti = j
	
	print starti
	
	return ids, users, numusers.astype(np.float32)