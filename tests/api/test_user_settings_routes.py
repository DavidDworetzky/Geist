from app.api.v1.endpoints.user_settings import router


def test_user_settings_api_only_exposes_the_current_workspace():
    routes = {(route.path, tuple(sorted(route.methods or ()))) for route in router.routes}

    assert ("/", ("GET",)) in routes
    assert ("/", ("PUT",)) in routes
    assert ("/reset", ("POST",)) in routes
    assert all("{user_id}" not in path for path, _methods in routes)
