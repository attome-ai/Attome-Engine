#include "ATMEngineUI.h"


float AnimationSystem::getAnimationPropertyValue(UIElement* element, const std::string& property) {
    if (!element) return 0.0f;

    // Example implementation for a few common properties
    if (property == "x") {
        return element->getPosition().x;
    }
    else if (property == "y") {
        return element->getPosition().y;
    }
    else if (property == "width") {
        return element->getBounds().w;
    }
    else if (property == "height") {
        return element->getBounds().h;
    }
    else if (property == "rotation") {
        return element->getRotation();
    }
    else if (property == "opacity") {
        return element->getOpacity();
    }

    return 0.0f;
}

void AnimationSystem::applyAnimationValue(UIElement* element, const std::string& property, float value) {
    if (!element) return;

    // Example implementation for a few common properties
    if (property == "x") {
        element->setPosition(value, element->getPosition().y);
    }
    else if (property == "y") {
        element->setPosition(element->getPosition().x, value);
    }
    else if (property == "width") {
        element->setSize(value, element->getBounds().h);
    }
    else if (property == "height") {
        element->setSize(element->getBounds().w, value);
    }
    else if (property == "rotation") {
        element->setRotation(value);
    }
    else if (property == "opacity") {
        element->setOpacity(value);
    }
}

// In implementation file
TextCache* g_textCache = nullptr;