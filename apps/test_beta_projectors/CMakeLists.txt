add_executable(sirius.test.beta sirius.test.beta.cpp)
target_link_libraries(sirius.test.beta PRIVATE sirius sirius::filesystem)
install(TARGETS sirius.test.beta RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
set_property(TARGET sirius.test.beta PROPERTY POSITION_INDEPENDENT_CODE OFF)
