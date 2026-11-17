// server-temporal.h — exhaustive temporal forensic audio+video analysis endpoint
// (see server-temporal.cpp). POST /v1/beehive/temporal returns the validated
// two-frame-group window plan for N fresh-context evidence calls + synthesis.
#pragma once

#include "server-beehive.h"
#include "server-http.h"

// Execute /v1/beehive/temporal: validates the request, returns the window plan.
server_http_res_ptr beehive_temporal_handler(server_beehive & bee, const server_http_req & req);